# =============================================================================
# USO:
#   Execute diretamente:  test_gpu_usage.py
#
# NOTA:
#   Esse é um arquivo para testar se o programa está reconhecendo
#   sua placa de vídeo(GPU) para uso durante o treinamento dos
#   modelos utilizados.
# =============================================================================


import tensorflow as tf

print("Versão do TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPU detectada! {len(gpus)} placa(s) disponível(is).")
    for gpu in gpus:
        print(f"   - {gpu.name}")
else:
    print("Nenhuma GPU detectada. O modelo vai rodar na CPU (mais lento).")