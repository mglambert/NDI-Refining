FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY ./external_code/xQSM/python/xQSM.py .
COPY ./external_code/xQSM/python/xQSM_blocks.py .
COPY api.py .
COPY custom_inference.py .

RUN apt-get update && apt-get install -y wget

RUN wget -O xQSM_invivo_withNoiseLayer.pth "https://www.dropbox.com/scl/fo/rk8si4esbxyqzsei53krw/AJL4GxboeltLIaULEH8bqIo/checkpoints/xQSM_invivo_withNoiseLayer.pth?rlkey=2f0n40mte1y78vgn5jbcmu7yb&dl=1"
RUN #wget -O xQSM_invivo.pth "https://www.dropbox.com/scl/fo/rk8si4esbxyqzsei53krw/AN8dODvihahB_zTNuNgy-oQ/checkpoints/xQSM_invivo.pth?rlkey=2f0n40mte1y78vgn5jbcmu7yb&dl=1"

RUN pip install Flask scipy

WORKDIR .

EXPOSE 5000

CMD ["python", "api.py"]





# https://www.dropbox.com/scl/fo/rk8si4esbxyqzsei53krw/AJL4GxboeltLIaULEH8bqIo/checkpoints/xQSM_invivo_withNoiseLayer.pth?rlkey=2f0n40mte1y78vgn5jbcmu7yb&e=1&st=o191wv7t&dl=0