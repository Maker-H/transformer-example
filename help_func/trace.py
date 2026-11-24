r"""코드가 실행되는 흐름을 실제 숫자로 따라가 보려고 찍는 로그 도우미입니다.
LOG를 False로 바꾸면 로그가 전부 꺼지고 계산 결과는 그대로예요.

로그 모드는 두 가지예요. 아래 LOG_MODE 값을 바꾸거나, 실행할 때 환경변수 TRANSFORMER_LOG로 정해요.
    간단 모드 "simple": 단계마다 "N번 한 줄 요약" 한 줄과 "N번 결과" 한 줄만 찍어요. (기본값)
    자세히 모드 "detail": 설명, 왜, 계산 과정, 다른 단어 결과까지 전부 찍어요.
PowerShell 예:  $env:TRANSFORMER_LOG="detail"; .venv\Scripts\python.exe inference_test.py
다시 간단 모드: Remove-Item Env:TRANSFORMER_LOG  (또는 $env:TRANSFORMER_LOG="simple")
모드는 로그를 보여 주는 양만 바꾸고, 계산 결과는 그대로예요.
"""

import inspect
import logging
import os
import re
import sys

import numpy as np

LOG = True

# 로그 모드예요. "simple"이면 간단 모드, "detail"이면 자세히 모드예요.
LOG_MODE = "simple"
# 환경변수 TRANSFORMER_LOG가 있으면 그 값으로 덮어써요. 예: TRANSFORMER_LOG=detail
if os.environ.get("TRANSFORMER_LOG", "").strip().lower() in ("simple", "detail"):
    LOG_MODE = os.environ["TRANSFORMER_LOG"].strip().lower()

# log(..., level=...)에 쓰는 표시예요.
#   SIMPLE: 간단 모드에서도 꼭 보여 줄 줄 (예: 레이어 사이 입력, memory)
#   DETAIL: 자세히 모드에서만 보여 줄 줄
#   표시가 없으면(None) 간단 모드에서 자동으로 골라요. 단계 안에서는 "한 줄 요약" 줄,
#   이번 단계의 "N번 결과" 줄, "블록이 끝났어요" 줄만 남기고 나머지는 숨겨요.
SIMPLE = "simple"
DETAIL = "detail"

# 블록마다 단계 순서예요. (화면에 보일 이름, 단계 제목에 들어 있는 말) 묶음이에요.
# 블록이 시작할 때 "순서: 정규화 > ..." 한 줄로 찍고, 단계 제목 앞에 "| 2/6 Q,K,V 만들기"처럼 지금 위치를 붙여요.
INPUT_STEPS = [
    ("임베딩", ("임베딩:",)),
    ("위치 인코딩", ("위치 정보",)),
]
ENCODER_SELF_STEPS = [
    ("정규화(LayerNorm)", ("LayerNorm",)),
    ("Q,K,V 만들기(Linear)", ("query_projection", "key_projection", "value_projection")),
    ("헤드 나누기", ("view로 쪼개기",)),
    ("헤드별 점수, softmax, V 섞기", (" attention:",)),
    ("헤드 합치고 W_O", ("head 합치기", "output_projection")),
    ("잔차 더하기", ("잔차 연결",)),
]
DECODER_SELF_STEPS = [
    ("정규화(LayerNorm)", ("LayerNorm",)),
    ("Q,K,V 만들기(Linear)", ("query_projection", "key_projection", "value_projection")),
    ("헤드 나누기", ("view로 쪼개기",)),
    ("헤드별 점수, 마스크, softmax, V 섞기", (" attention:",)),
    ("헤드 합치고 W_O", ("head 합치기", "output_projection")),
    ("잔차 더하기", ("잔차 연결",)),
]
SOURCE_STEPS = [
    ("재료 준비(Q는 블록 1 결과, K와 V는 memory)", ("재료 준비",)),
    ("정규화(LayerNorm)", ("LayerNorm",)),
    ("Q,K,V 만들기(Linear)", ("query_projection", "key_projection", "value_projection")),
    ("헤드 나누기", ("view로 쪼개기",)),
    ("헤드별 점수, softmax, V 섞기", (" attention:",)),
    ("헤드 합치고 W_O", ("head 합치기", "output_projection")),
    ("잔차 더하기", ("잔차 연결",)),
]
FEED_FORWARD_STEPS = [
    ("정규화(LayerNorm)", ("LayerNorm",)),
    ("넓히기(Linear)", ("expand_layer",)),
    ("ReLU", ("ReLU",)),
    ("줄이기(Linear)", ("shrink_layer",)),
    ("잔차 더하기", ("잔차 연결",)),
]
# main.py처럼 멀티헤드 어텐션만 따로 돌릴 때 쓰는 순서예요. 정규화와 잔차 더하기가 없어요.
ATTENTION_ONLY_STEPS = ENCODER_SELF_STEPS[1:-1]
GENERATOR_STEPS = [
    ("점수 만들기(Linear)", ("Generator output_projection",)),
    ("확률로 바꾸기(log_softmax)", ("log_softmax",)),
]

# 지금 블록의 단계 순서예요. place가 정해 줘요.
_steps = None
# 마지막으로 순서 지도를 찍은 블록 이름이에요. 같은 블록 지도를 두 번 찍지 않으려고 써요.
_map_shown = None
# 간단 모드에서 쓰는 값이에요. 지금 번호 단계 안인지, 이번 단계 결과 줄을 이미 찍었는지, 단계 머리 표시예요.
_in_section = False
_result_shown = False
_position = ""
# 이번 단계가 처음 만든 값의 역할 이름이에요. 간단 모드 결과 줄에 "(LayerNorm 결과)"처럼 붙여요.
_section_key = None
# 방금 순서 지도를 찍었는지예요. 지도 바로 아래에는 빈 줄을 넣지 않으려고 써요.
_fresh_map = False

# 지금 몇 단계 안쪽에서 실행 중인지 세는 값이에요. 안쪽일수록 로그가 더 들여써져요.
_depth = 0


# 로그에 찍을 단어 이름이에요. main.py에서 바꿀 수 있어요.
WORD_NAMES = ["A", "B", "C"]

# 처음부터 끝까지 따라갈 단어 번호예요. 0이면 단어A를 따라가요.
FOLLOW = 0

# 단계 번호를 세는 값이에요.
_section = 0


# 지금 모델의 어느 부분을 지나는 중인지 적어 두는 값이에요. 예: "인코더 레이어 1 / self-attention 블록"
# 단계 제목 앞에 붙어서 어느 파일, 어느 레이어인지 보여줘요.
WHERE = ""

# True면 다른 단어 결과를 한 줄로 몰아서 찍고, 표 W의 줄과 계산식을 한 줄에 같이 찍어서 로그를 짧게 해요.
# inference_test.py처럼 모델 전체를 돌릴 때 켜요.
COMPACT = False

# Q를 만드는 쪽 단어 이름과 K, V를 만드는 쪽 단어 이름이에요. None이면 WORD_NAMES를 써요.
# 디코더의 source attention에서는 Q는 디코더 단어, K와 V는 인코더 단어(memory)라서 따로 둬요.
Q_NAMES = None
KV_NAMES = None


# 단계마다 만든 값을 기억해 두는 곳이에요. 키는 값의 역할 이름(예: "셀프 어텐션 출력"), 값은 그 값을 만든 단계 번호예요.
# 로그에서 값을 "12번 결과_A"처럼 만든 단계 번호로 부르려고 써요. 계산에는 영향이 없어요.
_made = {}

# 지금 단계에서 이미 (역할)을 붙여 준 키예요. 같은 단계에서 두 번째부터는 (역할)을 빼서 줄을 짧게 해요.
_hinted = set()


def result(key, word):
    """이번 단계가 만든 값의 이름이에요. 예: "12번 결과_A".
    뒤 단계가 key로 찾을 수 있게, 이 값을 만든 단계 번호를 기억해 둬요."""
    global _section_key
    _made[key] = _section
    if _section_key is None:
        _section_key = key
    return f"{_section}번 결과_{word}"


def ref(key, word):
    """앞 단계가 만든 값의 이름이에요. 이번 단계에서 처음 쓸 때만 뒤에 (역할)을 붙여요.
    예: "11번 결과_A (셀프 어텐션 출력)". 만든 단계가 없는 값(처음 입력 x 등)은 "x_A"처럼 원래 이름을 써요."""
    number = _made.get(key)
    if number is None:
        return f"{key}_{word}"
    name = f"{number}번 결과_{word}"
    if key not in _hinted:
        _hinted.add(key)
        name += f" ({key})"
    return name


def name_of(key, word):
    """앞 단계가 만든 값의 이름 "47번 결과_시작"이에요. ref와 달리 (역할)을 붙이지 않고, 붙인 기록도 남기지 않아요."""
    number = _made.get(key)
    return f"{key}_{word}" if number is None else f"{number}번 결과_{word}"


def step_of(key):
    """key 값을 만든 단계 번호예요. 아직 없으면 None이에요."""
    return _made.get(key)


def _drop_files(text):
    """"디코더 레이어 1 (decoder_layer.py) / 블록 1"에서 "(파일 이름)"을 빼요."""
    return re.sub(r" \([^()]*\.py[^()]*\)", "", text)


def _step_text(title):
    """지금 블록 순서에서 이 단계가 몇 번째인지예요. 예: " | 2/6 Q,K,V 만들기(Linear)". 모르면 빈 문자열이에요."""
    if not _steps:
        return ""
    for index, (name, words) in enumerate(_steps):
        if any(word in title for word in words):
            return f" | {index + 1}/{len(_steps)} {name}"
    return ""


def section(title, file=None):
    """[1. 어디 | 몇 번째 단계 / 제목 (파일)] 처럼 번호를 붙여서 단계 제목을 찍어요.
    간단 모드에서는 제목 줄을 따로 찍지 않고, "한 줄 요약" 줄 앞에 [어디 | 몇 번째 단계]를 붙여요."""
    global _section, _in_section, _result_shown, _position, _section_key, _fresh_map
    _section += 1
    _hinted.clear()
    step_text = _step_text(title)
    where = f"{WHERE}{step_text} / " if WHERE else ""
    where_file = f" ({file})" if file else ""
    log("", level=DETAIL)
    log(f"[{_section}. {where}{title}{where_file}]", level=DETAIL)
    _position = f"{_drop_files(WHERE)}{step_text}" if WHERE else title.split(":")[0]
    # head마다 따로 찍는 단계는 어느 head인지도 붙여요. 예: "..., head0"
    if title.startswith("head") and " attention:" in title:
        _position += f", {title.split(' ')[0]}"
    # 간단 모드에서는 단계마다 빈 줄 하나, 위치 줄 하나를 찍고 그 아래에 값만 찍어요.
    if LOG_MODE == "simple":
        if not _fresh_map:
            log("", level=SIMPLE)
        log(f"[{_position}] {_section}번", level=SIMPLE)
    _fresh_map = False
    _in_section = True
    _result_shown = False
    _section_key = None


def matrix(title, rows, names):
    """Q, K, V처럼 단어마다 한 줄씩 있는 값을 행렬로 찍어요. 줄마다 단어 이름을 붙이고 모양도 써요.
    간단 모드와 자세히 모드 모두에서 보여요. 숫자는 받은 값을 그대로 찍기만 해요."""
    global _result_shown
    _result_shown = True
    log(f"  {title} (단어 {len(rows)}개 x {len(rows[0])}칸)", level=SIMPLE)
    for name, row in zip(names, rows):
        label = f"단어{name}" if str(name).isascii() else str(name)
        log(f"    {label} {fmt(row)}", level=SIMPLE)


def _pick(names, index):
    return names[index] if index < len(names) else f"단어{index}"


def word_name(index):
    """Q 쪽(지금 계산 중인 문장) 단어 이름이에요."""
    return _pick(Q_NAMES if Q_NAMES is not None else WORD_NAMES, index)


def kv_name(index):
    """K, V 쪽 단어 이름이에요. 따로 정하지 않았으면 Q 쪽 이름과 같아요."""
    if KV_NAMES is not None:
        return _pick(KV_NAMES, index)
    return word_name(index)


def word_label(name):
    """로그 문장에 쓰는 단어 이름이에요. A면 "단어A", 시작이면 "시작 단어"처럼 읽기 좋게 붙여요."""
    if str(name).isascii() or str(name).startswith("단어"):
        return name if str(name).startswith("단어") else f"단어{name}"
    return f"{name} 단어"


class place:
    """with place("인코더 레이어 1"): 안에서 찍는 단계 제목 앞에 그 위치를 붙여요.
    steps에 블록의 단계 순서를 주면, 들어갈 때 "[어디] 순서: 정규화 > ..." 한 줄을 먼저 찍어요."""

    def __init__(self, title, steps=None):
        self.title = title
        self.steps = steps

    def __enter__(self):
        global WHERE, _steps, _map_shown, _in_section, _fresh_map
        self.saved = WHERE
        self.saved_steps = _steps
        WHERE = f"{WHERE} / {self.title}" if WHERE else self.title
        if self.steps:
            _steps = self.steps
            if _map_shown != WHERE:
                _map_shown = WHERE
                _in_section = False
                log("", level=SIMPLE)
                log(f"[{_drop_files(WHERE)}] 순서: " + " > ".join(name for name, _ in self.steps), level=SIMPLE)
                _fresh_map = True
        return self

    def __exit__(self, *exc):
        global WHERE, _steps
        WHERE = self.saved
        _steps = self.saved_steps
        return False


def add_formula(name_a, a, name_b, b, name_out, out, others=None):
    """잔차 연결 같은 칸끼리 더하기를 칸마다 식으로 보여줘요.
    이름 뒤의 (역할)은 첫 줄에만 찍고, 아래 줄에서는 빼서 짧게 써요."""
    log(f"  {name_out} = {name_a} + {name_b}  (같은 자리 칸끼리 더함)")
    log(f"    {_short(name_a)}   {fmt(a)}")
    log(f"    {_short(name_b)}   {fmt(b)}")
    terms = ", ".join(f"{x:.2f}+{y:.2f}={z:.2f}" for x, y, z in zip(np.ravel(a), np.ravel(b), np.ravel(out)))
    log(f"    칸마다: {terms}")
    log(f"  {_short(name_out)} = {fmt(out)}")
    if others:
        log("  다른 단어: " + ",  ".join(others))


def _short(name):
    """"11번 결과_A (셀프 어텐션 출력)"에서 뒤의 (역할)을 뺀 "11번 결과_A"를 돌려줘요."""
    return name.split(" (")[0]


def fmt(vector):
    """[1.234, 5.678] 처럼 소수 둘째 자리까지 짧게 보여줘요."""
    return "[" + ", ".join(f"{v:.2f}" for v in np.ravel(vector)) + "]"


# 파이썬 기본 로깅 도구예요. Java의 logger처럼 로그 앞에 [클래스.함수]를 붙여 줘요.
_logger = logging.getLogger("trace")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(where)-40s %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

# 로그 도우미 함수들이에요. 위치를 찾을 때 이 함수들은 건너뛰고, 실제로 로그를 부른 곳을 찾아요.
_HELPERS = {"log", "log_array", "section", "add_formula", "__enter__", "__exit__", "_simple_line", "matrix"}


def _caller_name():
    """log()를 부른 곳을 거슬러 올라가서 "클래스.함수" 이름을 만들어요."""
    frame = inspect.currentframe().f_back
    while frame and frame.f_code.co_filename == __file__ and frame.f_code.co_name in _HELPERS | {"_caller_name"}:
        frame = frame.f_back
    if frame is None:
        return ""
    func = frame.f_code.co_name
    local_vars = frame.f_locals
    if "self" in local_vars:
        owner = type(local_vars["self"]).__name__
    elif "cls" in local_vars and isinstance(local_vars["cls"], type):
        owner = local_vars["cls"].__name__
    else:
        qualname = getattr(frame.f_code, "co_qualname", "")
        owner = qualname.rsplit(".", 1)[0] if "." in qualname else os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
    return f"[{owner}.{func}]"


def _simple_line(text, level):
    """간단 모드에서 이 줄을 어떻게 찍을지 정해요. 숨길 줄이면 None을 돌려줘요."""
    global _in_section, _result_shown
    if level == SIMPLE:
        return text
    if level == DETAIL:
        return None
    stripped = text.strip()
    # [디코더 레이어 1 시작], [디코딩 시작] 같은 큰 제목은 그대로 보여 주고, 번호 단계 밖으로 나왔다고 표시해요.
    if text.startswith("["):
        # 번호 단계 바로 뒤에 오는 큰 제목은 앞에 빈 줄 하나를 넣어서 구분해요.
        blank = "\n" if _in_section else ""
        _in_section = False
        return blank + text
    if not _in_section:
        return text
    # 간단 모드에서는 한 줄 요약도 숨겨요. 자세히 모드에서는 위치 제목 바로 아래 줄에 그대로 나와요.
    if stripped.startswith("한 줄 요약:"):
        return None
    if "블록이 끝났어요" in stripped:
        return f"  {stripped}"
    if not _result_shown:
        # 따라가는 단어 이름이에요. 소스 어텐션의 K, V처럼 입력 문장 쪽 값이면 kv_name 쪽 이름(A)으로 찍혀요.
        words = {re.escape(str(name)) for name in (word_name(0), word_name(FOLLOW), kv_name(0), kv_name(FOLLOW))}
        found = re.search(rf"(?<!\d){_section}번 결과_({'|'.join(words)}) = (\[[^\]]*\])", text)
        if found:
            _result_shown = True
            role = f" ({_section_key})" if _section_key else ""
            return f"  {_section}번 결과_{found.group(1)}{role} = {found.group(2)}"
    return None


def log(message="", level=None):
    """[클래스.함수]를 앞에 붙여서 한 줄을 찍어요.
    level은 SIMPLE(간단 모드에서도 보임), DETAIL(자세히 모드에서만 보임), None(간단 모드에서 자동으로 고름) 중 하나예요."""
    if LOG:
        # 텐서는 tensor([...]) 대신 pretty()로 모양과 줄 번호를 붙여서 읽기 쉽게 찍어요.
        text = message.pretty() if hasattr(message, "pretty") else str(message)
        if LOG_MODE == "simple":
            # 번호 단계 안에서 찍는 텐서 전체는 간단 모드에서 숨겨요. 단계 밖(학습 예제 등)에서는 그대로 보여요.
            if hasattr(message, "pretty") and _in_section:
                level = level or DETAIL
            text = _simple_line(text, level)
            if text is None:
                return
        where = _caller_name()
        # 여러 줄짜리 값(텐서 등)도 줄마다 [클래스.함수]가 붙게 한 줄씩 찍어요.
        for line in (text.splitlines() or [""]):
            _logger.info("    " * _depth + line, extra={"where": where})


def to_array(value):
    """Tensor든 numpy 배열이든 리스트든 numpy 배열로 바꿔요."""
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def log_array(title, value):
    """제목과 모양(shape)을 찍고, 그 아래에 실제 숫자를 찍어요."""
    if not LOG:
        return
    array = to_array(value)
    log(f"{title}  모양 {tuple(array.shape)}", level=DETAIL)
    text = np.array2string(array, precision=3, suppress_small=True, max_line_width=110)
    for line in text.splitlines():
        log("    " + line, level=DETAIL)


class step:
    """with step("제목"): 으로 감싼 블록 안의 로그를 한 칸 더 들여써요."""

    def __init__(self, title):
        self.title = title

    def __enter__(self):
        global _depth
        log(f"[{self.title}]")
        _depth += 1
        return self

    def __exit__(self, *exc):
        global _depth
        _depth -= 1
        return False


class quiet:
    """with quiet(): 안에서는 log를 잠깐 꺼요. 블록이 끝나면 원래 상태로 돌려놔요. 계산에는 영향이 없어요."""

    def __enter__(self):
        global LOG
        self.saved = LOG
        LOG = False
        return self

    def __exit__(self, *exc):
        global LOG
        LOG = self.saved
        return False
