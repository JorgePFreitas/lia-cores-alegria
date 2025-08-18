import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

def extrair_cores_dominantes(img, n_cores=5, ignorar_branco=True):
    """
    Extrai cores dominantes usando K-Means
    """
    # Reshape para lista de pixels
    pixels = img.reshape(-1, 3)
    
    # Remover pixels brancos se solicitado
    if ignorar_branco:
        # Pixels que NÃO são quase brancos (< 240 em qualquer canal)
        nao_brancos = ~((pixels[:, 0] > 240) & (pixels[:, 1] > 240) & (pixels[:, 2] > 240))
        pixels = pixels[nao_brancos]
    
    if len(pixels) < n_cores:
        return [], []
    
    # K-Means para encontrar cores dominantes
    kmeans = KMeans(n_clusters=n_cores, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Cores dominantes (centroids)
    cores = kmeans.cluster_centers_.astype(int)
    
    # Contar quantos pixels cada cor representa
    labels = kmeans.labels_
    contagens = np.bincount(labels)
    percentuais = (contagens / len(pixels)) * 100
    
    return cores, percentuais

def analisar_cores_dataset(caminho_pasta):
    """
    Analisa cores dominantes de todas as imagens
    """
    pasta = Path(caminho_pasta)
    resultados = []
    
    print("Analisando cores dominantes...")
    
    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar imagem
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Extrair 5 cores dominantes
            cores, percentuais = extrair_cores_dominantes(img, n_cores=5)
            
            if len(cores) > 0:
                resultado = {
                    'nome': arquivo.name,
                    'cor_1': cores[0].tolist(),
                    'perc_1': round(percentuais[0], 1),
                    'cor_2': cores[1].tolist() if len(cores) > 1 else None,
                    'perc_2': round(percentuais[1], 1) if len(cores) > 1 else None,
                    'cor_3': cores[2].tolist() if len(cores) > 2 else None,
                    'perc_3': round(percentuais[2], 1) if len(cores) > 2 else None,
                    'cor_4': cores[3].tolist() if len(cores) > 3 else None,
                    'perc_4': round(percentuais[3], 1) if len(cores) > 3 else None,
                    'cor_5': cores[4].tolist() if len(cores) > 4 else None,
                    'perc_5': round(percentuais[4], 1) if len(cores) > 4 else None,
                }
                
                resultados.append(resultado)
        
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    return pd.DataFrame(resultados)


if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Analisar cores
    df_cores = analisar_cores_dataset(caminho)
    
    # Salvar
    df_cores.to_csv('cores_dominantes.csv', index=False)
    print(f"\n💾 Dados salvos: cores_dominantes.csv")