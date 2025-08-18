import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

def rgb_para_hsv(img):
    """Converte BGR para HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def classificar_cor_hsv(h, s, v):
    """
    Classifica cor apenas por HSV (sem conversão RGB)
    """
    # Filtros por saturação e valor primeiro
    if v < 50:
        return "Preto/Muito Escuro"
    if v > 200 and s < 30:
        return "Branco/Muito Claro"
    if s < 30:
        return "Cinza/Dessaturado"
    
    # Classificação por matiz (H)
    if h < 8 or h > 172:
        return "Vermelho"
    elif h < 22:
        return "Laranja"
    elif h < 38:
        return "Amarelo"
    elif h < 78:
        return "Verde"
    elif h < 125:
        return "Azul"
    elif h < 155:
        return "Violeta"
    else:
        return "Rosa/Magenta"

def extrair_cores_hsv(img, n_cores=5):
    """
    Extrai cores dominantes HSV sem conversão RGB
    """
    # Converter para HSV
    hsv_img = rgb_para_hsv(img)
    pixels_hsv = hsv_img.reshape(-1, 3)
    
    # Filtrar pixels válidos (não muito claros, não muito dessaturados)
    h, s, v = pixels_hsv[:, 0], pixels_hsv[:, 1], pixels_hsv[:, 2]
    mask_valido = (v < 240) & (s > 25) & (v > 40)
    pixels_validos = pixels_hsv[mask_valido]
    
    if len(pixels_validos) < n_cores:
        return [], []
    
    # K-Means
    kmeans = KMeans(n_clusters=n_cores, random_state=42, n_init=10)
    kmeans.fit(pixels_validos)
    
    # Resultados
    cores_hsv = kmeans.cluster_centers_
    labels = kmeans.labels_
    contagens = np.bincount(labels)
    percentuais = (contagens / len(pixels_validos)) * 100
    
    return cores_hsv, percentuais

def analisar_hsv(caminho_pasta):
    """
    Análise HSV sem confusão RGB
    """
    pasta = Path(caminho_pasta)
    resultados = []
    
    print("Analise usando HSV")
    
    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar imagem
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Extrair cores HSV
            cores_hsv, percentuais = extrair_cores_hsv(img, n_cores=5)
            
            if len(cores_hsv) > 0:
                
                resultado = {'nome': arquivo.name}
                
                for i, (hsv, perc) in enumerate(zip(cores_hsv, percentuais)):
                    h, s, v = hsv.astype(int)
                    cor_tipo = classificar_cor_hsv(h, s, v)
                    
                    # Salvar no resultado
                    resultado[f'cor_{i+1}'] = cor_tipo
                    resultado[f'hsv_{i+1}'] = f"[{h},{s},{v}]"
                    resultado[f'perc_{i+1}'] = round(perc, 1)
                
                resultados.append(resultado)
        
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Análise HSV pura
    df_hsv = analisar_hsv(caminho)
    
    
    # Salvar
    df_hsv.to_csv('hsv.csv', index=False)
    print(f"\n💾 HSV puro salvo: hsv.csv")