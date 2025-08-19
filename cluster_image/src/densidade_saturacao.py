import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

def analisar_densidade_saturacao(img):
    """
    Analisa densidade de saturação sem definir cores específicas
    """
    # Converter para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    total_pixels = img.shape[0] * img.shape[1]
    
    # Categorias por saturação e valor
    resultados = {}
    
    # 4. Pixels dessaturados (cinzas)
    mask_cinza = (s <= 20) & (v > 40) & (v < 200)
    cinzas = np.sum(mask_cinza)
    resultados['cinza'] = round((cinzas / total_pixels) * 100, 2)
    
    # 5. Pixels muito claros (brancos)
    mask_branco = (v >= 200) & (s <= 30)
    brancos = np.sum(mask_branco)
    resultados['branco'] = round((brancos / total_pixels) * 100, 2)
    
    # 6. Pixels muito escuros (pretos)
    mask_preto = (v <= 40)
    pretos = np.sum(mask_preto)
    resultados['preto'] = round((pretos / total_pixels) * 100, 2)

    pixel_n_branco = total_pixels - brancos

    # 1. Pixels muito saturados (coloridos vívidos)
    mask_vivido = (s > 150) & (v > 80)
    vividos = np.sum(mask_vivido)
    resultados['colorido_vivido'] = round((vividos / pixel_n_branco) * 100, 2)
    
    # 2. Pixels moderadamente saturados (coloridos suaves)
    mask_suave = (s > 80) & (s <= 150) & (v > 60) & ~mask_vivido
    suaves = np.sum(mask_suave)
    resultados['colorido_suave'] = round((suaves / pixel_n_branco) * 100, 2)
    
    # 3. Pixels pouco saturados (quase monocromático)
    mask_mono = (s > 20) & (s <= 80) & (v > 50) & ~mask_vivido & ~mask_suave
    mono = np.sum(mask_mono)
    resultados['quase_monocromatico'] = round((mono / pixel_n_branco) * 100, 2)
    
    # Métricas gerais
    resultados['saturacao_media'] = round(np.mean(s), 2)
    resultados['valor_medio'] = round(np.mean(v), 2)
    
    # Classificação geral
    total_colorido = resultados['colorido_vivido'] + resultados['colorido_suave']
    
    if total_colorido > 40:
        resultados['classificacao'] = "Muito Colorido"
    elif total_colorido > 20:
        resultados['classificacao'] = "Colorido"
    elif total_colorido > 10:
        resultados['classificacao'] = "Pouco Colorido"
    else:
        resultados['classificacao'] = "Monocromático"
    
    return resultados

def calcular_diversidade_cores(img):
    """
    Calcula diversidade sem definir cores específicas
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Filtrar apenas pixels coloridos
    mask_colorido = (s > 50) & (v > 50) & (v < 230)
    
    if np.sum(mask_colorido) < 100:
        return 0
    
    # Pegar valores H dos pixels coloridos
    h_coloridos = h[mask_colorido]
    
    # Calcular dispersão dos matizes (0-179)
    # Quanto mais espalhado, mais diverso
    if len(h_coloridos) > 0:
        std_h = np.std(h_coloridos)
        # Normalizar para 0-100
        diversidade = min((std_h / 60) * 100, 100)
        return round(diversidade, 1)
    
    return 0

def analisar_densidade_dataset(caminho_pasta):
    """
    Analisa densidade de saturação de todas as imagens
    """
    pasta = Path(caminho_pasta)
    resultados = []
    
    print("ANÁLISE DE DENSIDADE DE SATURAÇÃO")
    
    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar imagem
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Analisar densidade
            densidade = analisar_densidade_saturacao(img)
            diversidade = calcular_diversidade_cores(img)
            
            # Preparar resultado
            resultado = {
                'nome': arquivo.name,
                'classificacao': densidade['classificacao'],
                'colorido_vivido': densidade['colorido_vivido'],
                'colorido_suave': densidade['colorido_suave'],
                'total_colorido': densidade['colorido_vivido'] + densidade['colorido_suave'],
                'monocromatico': densidade['quase_monocromatico'],
                'cinza': densidade['cinza'],
                'branco': densidade['branco'],
                'preto': densidade['preto'],
                'saturacao_media': densidade['saturacao_media'],
                'valor_medio': densidade['valor_medio'],
                'diversidade_cores': diversidade
            }
            
            resultados.append(resultado)
            
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Análise de densidade
    df_densidade = analisar_densidade_dataset(caminho)
    print(df_densidade.head(13))
    
    # Salvar
    df_densidade.to_csv('densidade_saturacao.csv', index=False)
    print(f"\n💾 Salvo: densidade_saturacao.csv")