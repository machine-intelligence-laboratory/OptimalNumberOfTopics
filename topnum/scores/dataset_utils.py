import pandas as pd
from topicnet.cooking_machine.dataset import get_modality_vw


# TODO: move this to TopicNet Dataset
# from topicnet.cooking_machine import Dataset
# ==================================


def col_total_len(modality):
    return f'len_total{modality}'


def col_uniq_len(modality):
    return f'len_uniq{modality}'


def count_tokens_unigram(text):
    result_uniq, result_total = 0, 0
    for raw_token in text.split():
        token, _, count = raw_token.partition(":")
        count = int(count or 1)
        result_uniq += 1
        result_total += count

    return result_total, result_uniq


def count_tokens_raw_tokenized(text):
    data_split = text.split()
    return len(data_split), len(set(data_split))


def compute_document_details(demo_data, all_mods):
    columns = [col_total_len(m) for m in all_mods] + [col_uniq_len(m) for m in all_mods]
    token_count_df = pd.DataFrame(index=demo_data._data.index, columns=columns)

    if demo_data._small_data:
        is_raw_tokenized = not demo_data._data.vw_text.str.contains(":").any()
    else:
        small_num_documents = 10
        documents = list(demo_data._data.index)[:small_num_documents]  # TODO: maybe slow here
        small_subdata = demo_data._data.loc[documents, :].compute()
        is_raw_tokenized = not small_subdata.vw_text.str.contains(":").any()

        del documents
        del small_subdata

    for m in all_mods:
        local_columns = col_total_len(m), col_uniq_len(m)
        vw_copy = demo_data._data.vw_text.apply(lambda vw_string: get_modality_vw(vw_string, m))

        if is_raw_tokenized:
            data = vw_copy.apply(count_tokens_raw_tokenized)
        else:
            data = vw_copy.apply(count_tokens_unigram)

        if not demo_data._small_data:
            data = data.compute()

        token_count_df.loc[:, local_columns] = pd.DataFrame(
            data.to_list(), index=data.index, columns=local_columns
        )

    return token_count_df

# ==================================
