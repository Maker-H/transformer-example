import arr_fun as nn

from transformer.attention.attention import attention
from help_func.clones import clones
from help_func import trace
from help_func.trace import fmt, log, to_array, word_name, kv_name


class MultiHeadedAttention(nn.Module):
    """단어 벡터를 head 개수만큼 작은 조각으로 나눠서 조각마다 따로 attention을 한 뒤 다시 합침
    head마다 서로 다른 관점(예: 문법 관계, 의미 관계)으로 다른 단어를 참고할 수 있게 함
    """

    def __init__(self, num_heads, model_dim, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 단어 벡터를 head 개수로 똑같이 나눌 수 있어야 함
        assert model_dim % num_heads == 0

        # head 하나가 맡는 벡터 칸 수 
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads

        # 같은 모양의 Linear 4개를 만들어서 query, key, value, 출력에 하나씩 씀
        projections = clones(nn.Linear(model_dim, model_dim), 4)
        self.query_projection = projections[0]
        self.key_projection = projections[1]
        self.value_projection = projections[2]
        self.output_projection = projections[3]

        # 로그에서 어느 Linear인지 구분하려고 이름을 붙여요.
        self.query_projection.log_name = "query_projection (Q 만들기)"
        self.key_projection.log_name = "key_projection (K 만들기)"
        self.value_projection.log_name = "value_projection (V 만들기)"
        self.output_projection.log_name = "output_projection (head 합친 뒤 마지막 Linear)"
        self.query_projection.output_label = "Q"
        self.key_projection.output_label = "K"
        self.value_projection.output_label = "V"
        self.query_projection.output_alias = "Q"
        self.key_projection.output_alias = "K"
        self.value_projection.output_alias = "V"
        self.output_projection.input_label = "head 합친 값"
        self.output_projection.output_label = "멀티헤드 어텐션 출력"
        self._set_log_about(source=False)
        # K, V는 key, value 쪽 문장에서 만들어요. source attention에서는 이 문장이 인코더 결과(memory)예요.
        self.key_projection.name_side = "kv"
        self.value_projection.name_side = "kv"

        # 마지막 forward에서 계산한 attention 비율을 저장해 둠
        self.attention_weights = None
        self.dropout = nn.Dropout(probability=dropout_rate)


    def set_log_labels(self, query_input, key_value_input, output, source=False):
        """로그에서 들어오는 값과 나가는 값을 부르는 이름을 정해요. 계산에는 영향이 없어요.
        source가 True면 디코더의 source attention이라서, K와 V를 인코더 결과(memory)로 만든다고 설명해요."""
        self.query_projection.input_label = query_input
        self.key_projection.input_label = key_value_input
        self.value_projection.input_label = key_value_input
        self.output_projection.output_label = output
        self._set_log_about(source)

    def _set_log_about(self, source):
        """Linear 4개의 단계 제목 아래에 찍을 "무엇을, 왜" 설명이에요. 계산에는 영향이 없어요."""
        kv_from = "인코더 결과(memory)의 단어" if source else "같은 문장의 단어"
        self.query_projection.log_about = [
            "한 줄 요약: 단어마다 학습 가중치 표 W_Q를 곱해서 Q(무엇을 찾는지 나타내는 값)를 만들어요.",
            "왜: 뒤에서 이 Q를 다른 단어의 K와 비교해서, 어떤 단어를 얼마나 참고할지 점수를 매겨요.",
        ]
        self.key_projection.log_about = [
            f"한 줄 요약: {kv_from}마다 학습 가중치 표 W_K를 곱해서 K(어떤 정보를 가졌는지 나타내는 이름표)를 만들어요.",
            "왜: Q와 K를 곱한 값이 '이 단어를 얼마나 참고할지' 점수가 돼요.",
        ]
        self.value_projection.log_about = [
            f"한 줄 요약: {kv_from}마다 학습 가중치 표 W_V를 곱해서 V(실제로 넘겨줄 내용)를 만들어요.",
            "왜: 점수로 참고 비율이 정해지면, 그 비율만큼 실제로 섞어서 가져가는 정보가 V예요.",
        ]
        self.output_projection.log_about = [
            "한 줄 요약: head 합친 값에 학습 가중치 표 W_O를 곱해요. W_O도 Q, K, V를 만든 W와 같은 종류의 학습 가중치예요.",
            f"왜: 그냥 이어 붙이기만 하면 head마다 만든 정보가 칸만 나란히 있고 안 섞여요. W_O를 곱하면 결과 칸 하나하나가 {self.num_heads * self.head_dim}칸 전부를 참고해서 head들의 정보가 섞여요.",
        ]

    def forward(self, query, key, value, mask=None):
        return self._forward(query, key, value, mask)

    @staticmethod
    def _split_text(label, heads_tensor, whole_key):
        """view로 쪼갠 결과를 단어마다 한 줄 글로 만들어요. Q, K, V 세 번 똑같이 쓰는 모양이라 따로 뺐어요.
        글만 만들고 찍지는 않아요. 찍는 건 _forward 안의 log가 해서 로그 앞에 [MultiHeadedAttention._forward]가 붙어요."""
        split = to_array(heads_tensor)[0]
        name_of = word_name if label == "Q" else kv_name
        follow = trace.FOLLOW if trace.FOLLOW < split.shape[1] else 0
        lines = []
        for i in range(split.shape[1]):
            if trace.COMPACT and i != follow:
                continue
            whole = list(split[:, i].ravel())
            pieces = "   ".join(f"{label}_{name_of(i)}(head{h}) = {fmt(split[h, i])}" for h in range(split.shape[0]))
            lines.append(f"    {trace.ref(whole_key, name_of(i))} {fmt(whole)}  쪼개면  {pieces}")
        return "\n".join(lines)

    def _forward(self, query, key, value, mask=None):
        if mask is not None:
            # 모든 head에 같은 mask를 쓰려고 head 축 자리(1번 축)를 하나 추가함
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 예: 문장 1개, 단어 3개, model_dim = 4, num_heads = 2, head_dim = 2 (head 하나가 맡는 벡터 칸 수)
        # query 모양 (1, 3, 4)
        # [[[1, 2, 3, 4],       # 단어A
        #   [5, 6, 7, 8],       # 단어B
        #   [9, 10, 11, 12]]]   # 단어C

        # Linear를 통과해도 모양은 (1, 3, 4) 그대로이고 값만 바뀜
        # view로 단어마다 벡터 4칸을 2칸짜리 head 2개로 쪼갬, 모양 (1, 3, 2, 2)
        # 단어A [1, 2, 3, 4]는 head0 [1, 2], head1 [3, 4]가 됨
        # transpose(1, 2)로 head 축을 앞으로 옮김, 모양 (1, 2, 3, 2)
        # head0에는 단어A, B, C의 앞 2칸이, head1에는 뒤 2칸이 모여서 head마다 따로 attention을 계산할 수 있음
        query_heads = ...

        
        # (1, -1, 2, 2)
        shape = (batch_size, -1, self.num_heads, self.head_dim)

        # query를 weight와 곱하고 bias를 더함 -> 이때 shape(1, 3, 4)
        # view() -1은 자동으로 계산되어서 12개 숫자를 순서 그대로 (1, 3, 2, 2) 모양으로 바꿔줌
        # transpose() 축 순서 바꿔서 (1, 2, 3, 2) 모양으로 만듬        
        # 원래 한 줄: query_heads = self.query_projection(query).view(shape).transpose(1, 2)
        # 중간 값을 보려고 나눴고 계산은 같아요. key, value도 똑같이 나눴어요.
        # Linear 세 개를 먼저 통과시키고, 그다음에 쪼개기 설명과 결과를 Q, K, V 순서로 찍어요.
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)

        if trace.LOG:
            trace.section("view로 쪼개기: 4칸을 앞 2칸(head0), 뒤 2칸(head1)으로 나눔", "multi_headed_attention.py")
            log(f"  한 줄 요약: Q, K, V를 head마다 {self.head_dim}칸씩 나눠요. 숫자는 그대로고 나누기만 해요.")
            log("  왜: head마다 따로 attention을 해서, head마다 서로 다른 관점으로 다른 단어를 참고하게 하려고요.")
            trace.result("head로 쪼갠 Q, K, V", "")
            log(f"  아래 Q_A(head0)처럼 부르는 값이 이 {trace.step_of('head로 쪼갠 Q, K, V')}번 단계에서 쪼갠 조각이에요.")
        query_heads = query_projected.view(shape).transpose(1, 2)
        if trace.LOG:
            log(self._split_text("Q", query_heads, self.query_projection.output_label), level=trace.SIMPLE)
        key_heads = key_projected.view(shape).transpose(1, 2)
        if trace.LOG:
            log(self._split_text("K", key_heads, self.key_projection.output_label), level=trace.SIMPLE)
        value_heads = value_projected.view(shape).transpose(1, 2)
        if trace.LOG:
            log(self._split_text("V", value_heads, self.value_projection.output_label), level=trace.SIMPLE)
            if trace.COMPACT and to_array(key_heads).shape[2] > 1:
                log("    다른 단어의 K, V도 똑같이 앞 2칸은 head0, 뒤 2칸은 head1로 감")

        # 2) head마다 attention을 한꺼번에 계산함
        attention_output, self.attention_weights = attention(
            query_heads, key_heads, value_heads, mask=mask, dropout_layer=self.dropout
        )

        # 3) 나눴던 head들을 다시 합침
        #    transpose(1, 2)로 (배치, 단어 수, head 수, head_dim) 모양으로 되돌리고
        #    view로 head들을 이어 붙여 (배치, 단어 수, model_dim) 모양으로 만듦
        #    contiguous는 transpose 뒤에 view를 쓸 수 있도록 메모리 순서를 정리함
        if trace.LOG:
            trace.section("head 합치기: head0 결과 2칸 뒤에 head1 결과 2칸을 이어 붙임", "multi_headed_attention.py")
            log(f"  한 줄 요약: head마다 따로 만든 결과를 순서대로 이어 붙여서 다시 {self.num_heads * self.head_dim}칸 벡터 하나로 만들어요.")
            log("  왜: 다음 단계(W_O)에서 head들의 정보를 섞으려면, 먼저 한 벡터로 모아야 해요.")
        merged_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        if trace.LOG:
            merged = to_array(merged_output)[0]
            split = to_array(attention_output)[0]
            f = trace.FOLLOW
            me = word_name(f)
            merged_key = self.output_projection.input_label
            pieces = " 와 ".join(f"{trace.ref(f'head{h} attention 결과', me)} {fmt(split[h, f])}" for h in range(split.shape[0]))
            log(f"    {pieces}  붙이면  {trace.result(merged_key, me)} = {fmt(merged[f])}")
            others = [f"{trace.result(merged_key, word_name(i))} = {fmt(merged[i])}" for i in range(merged.shape[0]) if i != f]
            if trace.COMPACT and others:
                log("    다른 단어: " + ",  ".join(others))
            else:
                for line in others:
                    log(f"    {line}")

        # 합친 결과를 마지막 Linear에 통과시켜서 리턴
        final_output = self.output_projection(merged_output)
        if trace.LOG:
            me = word_name(trace.FOLLOW)
            log(f"  여기서 멀티헤드 어텐션이 끝났어요. 멀티헤드 어텐션의 출력은 {trace._short(trace.ref(self.output_projection.output_label, me))} 값이에요.")
        return final_output
