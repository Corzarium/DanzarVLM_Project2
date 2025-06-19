@echo off
echo Setting Hugging Face cache to E: drive to avoid C: drive space issues...

REM Set Hugging Face cache directories to E: drive
set HF_HOME=E:\huggingface_cache
set TRANSFORMERS_CACHE=E:\huggingface_cache\transformers
set HF_DATASETS_CACHE=E:\huggingface_cache\datasets
set HUGGINGFACE_HUB_CACHE=E:\huggingface_cache\hub

echo HF_HOME=%HF_HOME%
echo TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo HF_DATASETS_CACHE=%HF_DATASETS_CACHE%
echo HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%

echo Starting DanzarVLM...
python DanzarVLM.py 