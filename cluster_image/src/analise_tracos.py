import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy import ndimage

def analisar_tracos_desenho_corrigido(img):
    """
    Análise CORRIGIDA de traços com algoritmos melhorados
    """
    # Converter para grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Inverter se necessário (fundo claro, traços escuros)
    if np.mean(gray) > 127:
        gray = 255 - gray
    
    total_pixels = gray.shape[0] * gray.shape[1]
    
    # Criar máscara de traços (pixels significativos)
    mask_tracos = gray > 20  # Threshold mais baixo para capturar traços leves
    
    # === 1. ANÁLISE DE ESPESSURA CORRIGIDA ===
    # Usar DISTANCE TRANSFORM - mede distância real até borda mais próxima
    
    # Binarizar imagem
    binary = (gray > 20).astype(np.uint8)
    
    if np.sum(binary) > 100:  # Se há traços suficientes
        # Distance transform: cada pixel mostra distância até borda
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Espessura = 2 * distância máxima (raio até centro do traço)
        espessuras_locais = dist_transform[dist_transform > 0] * 2
        
        if len(espessuras_locais) > 0:
            espessura_media = round(np.mean(espessuras_locais), 2)
            espessura_max = round(np.max(espessuras_locais), 2)
            espessura_std = round(np.std(espessuras_locais), 2)
        else:
            espessura_media = espessura_max = espessura_std = 0.0
    else:
        espessura_media = espessura_max = espessura_std = 0.0
    
    # === 2. ANÁLISE DE CONTINUIDADE CORRIGIDA ===
    # Usar múltiplas estratégias para medir continuidade
    
    # 2.1 Canny com parâmetros ajustados para desenhos à mão
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # Suavizar antes do Canny
    edges = cv2.Canny(blur, 30, 80)  # Thresholds mais baixos
    
    # 2.2 Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filtrar contornos muito pequenos (ruído)
        contours_validos = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        
        num_segmentos = len(contours_validos)
        
        # Calcular comprimentos dos contornos
        comprimentos = [cv2.arcLength(cnt, False) for cnt in contours_validos]
        comprimento_total = sum(comprimentos)
        
        if num_segmentos > 0:
            comprimento_medio = round(comprimento_total / num_segmentos, 2)
        else:
            comprimento_medio = 0.0
            
        # 2.3 Métrica de conectividade: razão entre área real e área de contornos
        area_tracos = np.sum(mask_tracos)
        if comprimento_total > 0:
            conectividade = round((area_tracos / comprimento_total), 2)
        else:
            conectividade = 0.0
            
    else:
        num_segmentos = 0
        comprimento_total = comprimento_medio = conectividade = 0.0
    
    # === 3. ANÁLISE DE SUAVIDADE MELHORADA ===
    # Combinar múltiplas métricas de suavidade
    
    # 3.1 Laplaciano original (detecta rugosidade)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    rugosidade_laplacian = np.std(laplacian[mask_tracos]) if np.sum(mask_tracos) > 0 else 0
    
    # 3.2 Gradiente magnitude (detecta mudanças bruscas)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    rugosidade_gradiente = np.std(magnitude[mask_tracos]) if np.sum(mask_tracos) > 0 else 0
    
    # 3.3 Combinar métricas
    rugosidade_total = (rugosidade_laplacian + rugosidade_gradiente) / 2
    suavidade = round(1 / (1 + rugosidade_total / 50), 3)  # Normalização ajustada
    
    # === 4. ANÁLISE DE DENSIDADE REFINADA ===
    densidade_tracos = round((np.sum(mask_tracos) / total_pixels) * 100, 2)
    
    # Densidade local (variação espacial)
    # Dividir imagem em grid 4x4 e calcular densidade por região
    h, w = gray.shape
    grid_densities = []
    
    for i in range(4):
        for j in range(4):
            y1, y2 = (i * h // 4), ((i + 1) * h // 4)
            x1, x2 = (j * w // 4), ((j + 1) * w // 4)
            
            region = mask_tracos[y1:y2, x1:x2]
            densidade_local = np.sum(region) / (region.size) * 100
            grid_densities.append(densidade_local)
    
    variacao_densidade = round(np.std(grid_densities), 2)
    densidade_max_regiao = round(max(grid_densities), 2)
    
    # === 5. ANÁLISE DE PRESSÃO MELHORADA ===
    # Usar histograma adaptativo baseado na distribuição real
    
    pixels_tracos = gray[mask_tracos]
    
    if len(pixels_tracos) > 0:
        # Calcular percentis para thresholds adaptativos
        p25 = np.percentile(pixels_tracos, 25)
        p50 = np.percentile(pixels_tracos, 50)
        p75 = np.percentile(pixels_tracos, 75)
        
        # Categorizar pressão baseado em percentis
        pressao_fraca = np.sum((pixels_tracos >= 20) & (pixels_tracos < p25))
        pressao_media = np.sum((pixels_tracos >= p25) & (pixels_tracos < p75))
        pressao_forte = np.sum(pixels_tracos >= p75)
        
        total_pixels_tracos = len(pixels_tracos)
        
        pressao_fraca_pct = round((pressao_fraca / total_pixels_tracos) * 100, 2)
        pressao_media_pct = round((pressao_media / total_pixels_tracos) * 100, 2)
        pressao_forte_pct = round((pressao_forte / total_pixels_tracos) * 100, 2)
        
        intensidade_media = round(np.mean(pixels_tracos), 2)
        contraste_pressao = round(np.std(pixels_tracos), 2)
        
    else:
        pressao_fraca_pct = pressao_media_pct = pressao_forte_pct = 0.0
        intensidade_media = contraste_pressao = 0.0
        p25 = p50 = p75 = 0
    
    # === 6. ANÁLISE DE COMPLEXIDADE GEOMÉTRICA ===
    # Medir complexidade usando entropy e fractal dimension
    
    # 6.1 Entropy da imagem (medida de complexidade)
    if np.sum(mask_tracos) > 0:
        hist, _ = np.histogram(gray[mask_tracos], bins=32, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalizar
        hist = hist[hist > 0]  # Remover zeros
        entropy = -np.sum(hist * np.log2(hist))
        entropia_normalizada = round(entropy / 5, 3)  # Normalizar 0-1
    else:
        entropia_normalizada = 0.0
    
    # === CLASSIFICAÇÕES MELHORADAS ===
    
    # Classificação de espessura baseada em dados reais
    if espessura_media < 1.5:
        classificacao_espessura = "Traços Muito Finos"
    elif espessura_media < 3.0:
        classificacao_espessura = "Traços Finos"
    elif espessura_media < 5.0:
        classificacao_espessura = "Traços Médios"
    elif espessura_media < 8.0:
        classificacao_espessura = "Traços Grossos"
    else:
        classificacao_espessura = "Traços Muito Grossos"
    
    # Classificação de continuidade baseada em conectividade
    if conectividade > 15:
        classificacao_continuidade = "Muito Conectado"
    elif conectividade > 8:
        classificacao_continuidade = "Conectado"
    elif conectividade > 4:
        classificacao_continuidade = "Moderadamente Conectado"
    elif conectividade > 2:
        classificacao_continuidade = "Pouco Conectado"
    else:
        classificacao_continuidade = "Fragmentado"
    
    # Classificação de densidade
    if densidade_tracos > 25:
        classificacao_densidade = "Muito Denso"
    elif densidade_tracos > 15:
        classificacao_densidade = "Denso"
    elif densidade_tracos > 8:
        classificacao_densidade = "Moderado"
    elif densidade_tracos > 3:
        classificacao_densidade = "Esparso"
    else:
        classificacao_densidade = "Muito Esparso"
    
    resultados = {
        'espessura_media': espessura_media,
        'espessura_max': espessura_max,
        'espessura_std': espessura_std,
        'num_segmentos': num_segmentos,
        'comprimento_total': round(comprimento_total, 1),
        'comprimento_medio': comprimento_medio,
        'conectividade': conectividade,
        'suavidade': suavidade,
        'densidade_tracos': densidade_tracos,
        'variacao_densidade': variacao_densidade,
        'densidade_max_regiao': densidade_max_regiao,
        'pressao_forte_pct': pressao_forte_pct,
        'pressao_media_pct': pressao_media_pct,
        'pressao_fraca_pct': pressao_fraca_pct,
        'intensidade_media': intensidade_media,
        'contraste_pressao': contraste_pressao,
        'entropia_normalizada': entropia_normalizada,
        'thresholds_pressao': f"Fraca:<{p25:.0f}, Média:{p25:.0f}-{p75:.0f}, Forte:>{p75:.0f}",
        'classificacao_espessura': classificacao_espessura,
        'classificacao_continuidade': classificacao_continuidade,
        'classificacao_densidade': classificacao_densidade
    }
    
    return resultados
    
def analisar_tracos_dataset_corrigido(caminho_pasta):
    """
    Processa dataset com algoritmos corrigidos
    """
    pasta = Path(caminho_pasta)
    resultados = []
    
    print("ANÁLISE DE TRAÇOS - VERSÃO CORRIGIDA")
    print("=" * 50)
    
    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar imagem
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Analisar com algoritmos corrigidos
            tracos = analisar_tracos_desenho_corrigido(img)
            
            # Preparar resultado
            resultado = {
                'nome': arquivo.name,
                'classificacao_espessura': tracos['classificacao_espessura'],
                'classificacao_continuidade': tracos['classificacao_continuidade'],
                'classificacao_densidade': tracos['classificacao_densidade'],
                'espessura_media': tracos['espessura_media'],
                'espessura_max': tracos['espessura_max'],
                'espessura_std': tracos['espessura_std'],
                'densidade_tracos': tracos['densidade_tracos'],
                'variacao_densidade': tracos['variacao_densidade'],
                'num_segmentos': tracos['num_segmentos'],
                'conectividade': tracos['conectividade'],
                'comprimento_total': tracos['comprimento_total'],
                'suavidade': tracos['suavidade'],
                'pressao_forte_pct': tracos['pressao_forte_pct'],
                'pressao_media_pct': tracos['pressao_media_pct'],
                'pressao_fraca_pct': tracos['pressao_fraca_pct'],
                'intensidade_media': tracos['intensidade_media'],
                'contraste_pressao': tracos['contraste_pressao'],
                'entropia_normalizada': tracos['entropia_normalizada'],
                'thresholds_pressao': tracos['thresholds_pressao']
            }
            
            resultados.append(resultado)
            print(f"✅ {arquivo.name} - Espessura: {tracos['espessura_media']:.1f}px, Conectividade: {tracos['conectividade']:.1f}")
            
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Análise corrigida
    df_tracos = analisar_tracos_dataset_corrigido(caminho)
    
    if not df_tracos.empty:
        print("\n" + "=" * 50)
        print("=== RESULTADOS ALGORITMOS CORRIGIDOS ===")
        print("=" * 50)
        
        
        # Salvar resultados corrigidos
        df_tracos.to_csv('analise_tracos_corrigida.csv', index=False)
        

    else:
        print("❌ Nenhuma imagem processada!")