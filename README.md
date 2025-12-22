vllm serve crestai/spark-tts-nexvox --port 8002 --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-chunked-prefill --enable-prefix-caching --trust-remote-code

pip install python-multipart