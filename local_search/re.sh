#!/bin/bash

# 设置变量
save_path="/your_path/wiki25"  # ⚠️ 请修改为实际路径
index_file="$save_path/e5_Flat.index"
corpus_file="$save_path/wiki25_new.jsonl"
retriever_name="e5"
retriever_path="/your_path/wiki25/e5-base-v2-main"

# 启动检索服务

python /your_path/local_search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu