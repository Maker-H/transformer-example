import math

import arr_fun as torch
from help_func import trace
from help_func.trace import fmt, log, to_array, word_name, kv_name


# additive attention, dot-project attention 방식은 비슷한데 dk 가산하는 방식만 다름 
def attention(query, key, value, mask=None, dropout_layer=None):
    """각 단어가 다른 단어들을 얼마나 참고할지 비율을 정하고, 그 비율대로 value를 섞어서 리턴"""
    
    "Compute 'Scaled Dot Product Attention'"
    dim_key = query.size(-1)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    attention_weights = attention_scores.softmax(dim=-1)

    if dropout_layer is not None:
        attention_weights = dropout_layer(attention_weights)
    output = torch.matmul(attention_weights, value)

    if trace.LOG:
        _log_each_head(query, key, value, attention_scores, attention_weights, output, mask)
    return output, attention_weights


def _log_each_head(query, key, value, scores, weights, output, mask=None):
    """head마다, 따라가는 단어 하나가 무슨 일을 겪는지 보여줘요. (배치 0번만)"""
    q, k, v = to_array(query)[0], to_array(key)[0], to_array(value)[0]
    s, w, o = to_array(scores)[0], to_array(weights)[0], to_array(output)[0]
    num_heads, num_words, num_keys = w.shape[0], w.shape[1], w.shape[2]
    i = trace.FOLLOW if trace.FOLLOW < num_words else 0
    me = word_name(i)
    dim = q.shape[-1]
    root = math.sqrt(dim)
    # 마스크에서 따라가는 단어의 줄만 꺼내요. 1이면 보이고 0이면 가려요.
    visible = [True] * num_keys
    if mask is not None:
        m = to_array(mask)
        m = m.reshape((-1,) + m.shape[-2:])[0]
        row = m[i] if m.shape[0] > 1 else m[0]
        visible = [bool(row[j]) for j in range(num_keys)]
    key_names = ", ".join(f"K_{kv_name(j)}" for j in range(num_keys))

    for head in range(num_heads):
        h = f"(head{head})"
        trace.section(f"head{head} attention: {trace.word_label(me)}가 다른 단어를 얼마나 참고할지", "attention.py")
        log(f"  한 줄 요약: 점수로 누구를 얼마나 참고할지 정하고, softmax로 그 점수를 합이 100%인 비율로 바꾸고, 그 비율만큼 V를 섞어서 {trace.word_label(me)}의 새 값을 만들어요.")
        log(f"  왜: 점수만 구하면 {trace.word_label(me)}는 하나도 안 바뀌어요. 비율대로 V를 실제로 섞어야 다른 단어의 정보가 {trace.word_label(me)}에 들어와요.")
        split_step = trace.step_of("head로 쪼갠 Q, K, V")
        where_from = f" ({split_step}번 단계에서 쪼갠 조각)" if split_step is not None else ""
        log(f"  이 head가 쓰는 값{where_from}")
        log(f"    {trace.word_label(me)}의 Q: Q_{me}{h} = {fmt(q[head, i])}")
        for j in range(num_keys):
            log(f"    {trace.word_label(kv_name(j))}로 만든 K: K_{kv_name(j)}{h} = {fmt(k[head, j])}   V: V_{kv_name(j)}{h} = {fmt(v[head, j])}")
        if mask is not None:
            seen = ", ".join(f"{kv_name(j)} {'보임' if visible[j] else '가림'}" for j in range(num_keys))
            log(f"  마스크 ({trace.word_label(me)}가 볼 수 있는 단어): {seen}")
        if num_keys == 1:
            log(f"  볼 수 있는 단어가 {kv_name(0)} 하나뿐이라 점수와 상관없이 비율은 100%예요.")
            log(f"  그래서 결과는 V_{kv_name(0)}{h} 그대로예요: 1.00*{fmt(v[head, 0])}")
            log(f"  {trace.result(f'head{head} attention 결과', me)} = {fmt(o[head, i])}   (head{head}이 만든 {trace.word_label(me)}의 새 값)")
            continue
        log(f"  점수: Q와 K를 곱해서, {trace.word_label(me)}가 각 단어를 얼마나 참고할지 숫자로 매겨요. Q와 K가 비슷할수록 점수가 커요.")
        log(f"    점수 = Q_{me}{h} 와 {key_names} 를 하나씩 칸끼리 곱해서 더함, 그리고 sqrt({dim}) = {root:.2f} 로 나눔")
        for j in range(num_keys):
            other = kv_name(j)
            terms = " + ".join(f"{q[head, i, c]:.2f}*{k[head, j, c]:.2f}" for c in range(dim))
            if visible[j]:
                log(f"    Q_{me}{h} {fmt(q[head, i])} 와 K_{other}{h} {fmt(k[head, j])}: ({terms}) / {root:.2f} = {s[head, i, j]:.2f}")
            else:
                log(f"    K_{other}{h}: 마스크로 가려서 점수를 아주 작은 값(-1000000000)으로 바꿈")
        exps = [math.exp(s[head, i, j]) if visible[j] else 0.0 for j in range(num_keys)]
        total = sum(exps)
        log("  softmax: 위 점수를 합이 100%인 참고 비율로 바꿔요. 점수가 클수록 비율이 커요.")
        log("    softmax = e^점수 / (e^점수 전부 더한 값)")
        log("      " + ", ".join(f"e^{s[head, i, j]:.2f} = {exps[j]:.2f}" if visible[j] else f"{kv_name(j)} 가림 = 0.00" for j in range(num_keys)) + f",  합 {total:.2f}")
        if trace.COMPACT:
            log("      비율: " + ",  ".join(f"{kv_name(j)} {exps[j]:.2f}/{total:.2f} = {w[head, i, j]:.2f} ({w[head, i, j] * 100:.0f}%)" for j in range(num_keys)))
        else:
            for j in range(num_keys):
                log(f"      {kv_name(j)} 비율 = {exps[j]:.2f} / {total:.2f} = {w[head, i, j]:.2f} ({w[head, i, j] * 100:.0f}%)")
        log("  V 섞기: 그 비율만큼 단어마다 V(내용)를 가져와서 칸마다 더해요. 비율이 큰 단어의 V가 결과에 많이 들어가요.")
        log("    V 섞기 = 비율 * V 를 칸마다 더함")
        for c in range(v.shape[-1]):
            terms = " + ".join(f"{w[head, i, j]:.2f}*{v[head, j, c]:.2f}(V_{kv_name(j)})" for j in range(num_keys))
            log(f"      {c}번째 칸 = {terms} = {o[head, i, c]:.2f}")
        log(f"  {trace.result(f'head{head} attention 결과', me)} = {fmt(o[head, i])}   (head{head}이 다른 단어를 참고해서 만든 {trace.word_label(me)}의 새 값)")
