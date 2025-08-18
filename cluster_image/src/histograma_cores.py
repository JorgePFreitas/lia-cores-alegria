import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

def definir_faixas_cores():
    """
    Define faixas de cores HSV para classificação
    """
    faixas = {
        'Vermelho': [(0, 8), (172, 179)],    # H: 0-8 e 172-179
        'Laranja': [(8, 22)],                # H: 8-22  
        'Amarelo': [(22, 38)],               # H: 22-38
        'Verde': [(38, 78)],                 # H: 38-78
        'Azul': [(78, 125)],                 # H: 78-125
        'Violeta': [(125, 155)],             # H: 125-155
        'Rosa': [(155, 172)]                 # H: 155-172
    }
    return faixas

def contar_pixels_por_cor(img):
    """
    Conta pixels por faixa de cor usando histograma
    """
    # Converter para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Filtrar pixels válidos (não muito claros/dessaturados)
    mask_valido = (s > 30) & (v > 50) & (v < 240)
    
    total_pixels_validos = np.sum(mask_valido)
    if total_pixels_validos == 0:
        return {}
    
    # Contar por faixa de cor
    faixas = definir_faixas_cores()
    contadores = {}
    
    for cor, intervalos in faixas.items():
        mask_cor = np.zeros_like(h, dtype=bool)
        
        for inicio, fim in intervalos:
            mask_cor |= (h >= inicio) & (h <= fim)
        
        # Combinar com mask válido
        mask_final = mask_cor & mask_valido
        pixels_cor = np.sum(mask_final)
        percentual = (pixels_cor / total_pixels_validos) * 100
        
        if percentual > 1:  # Só cores com mais de 1%
            contadores[cor] = round(percentual, 1)
    
    # Adicionar categorias especiais
    mask_preto = (v < 50) & (s > 20)
    mask_branco = (v > 200) & (s < 30)
    mask_cinza = (s < 30) & (v >= 50) & (v <= 200)
    
    total_todos_pixels = h.shape[0] * h.shape[1]
    
    preto_perc = (np.sum(mask_preto) / total_todos_pixels) * 100
    branco_perc = (np.sum(mask_branco) / total_todos_pixels) * 100
    cinza_perc = (np.sum(mask_cinza) / total_todos_pixels) * 100
    
    if preto_perc > 1:
        contadores['Preto'] = round(preto_perc, 1)
    if branco_perc > 1:
        contadores['Branco'] = round(branco_perc, 1)
    if cinza_perc > 1:
        contadores['Cinza'] = round(cinza_perc, 1)
    
    return contadores

def analisar_histograma_cores(caminho_pasta):
    """
    Analisa cores usando histograma direto
    """
    pasta = Path(caminho_pasta)
    resultados = []
    
    print("📊 ANÁLISE POR HISTOGRAMA DE CORES")

    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar imagem
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Contar cores
            cores_encontradas = contar_pixels_por_cor(img)
            
            print(f"\n📁 {arquivo.name}:")
            
            # Ordenar por percentual
            cores_ordenadas = sorted(cores_encontradas.items(), key=lambda x: x[1], reverse=True)
            
            resultado = {'nome': arquivo.name}
            
            for i, (cor, perc) in enumerate(cores_ordenadas[:5]):  # Top 5
                resultado[f'cor_{i+1}'] = cor
                resultado[f'perc_{i+1}'] = perc
            
            resultados.append(resultado)
            
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Análise por histograma
    df_hist = analisar_histograma_cores(caminho)
    
    # Salvar
    df_hist.to_csv('histograma_cores.csv', index=False)
    print(f"\n💾 Salvo: histograma_cores.csv")