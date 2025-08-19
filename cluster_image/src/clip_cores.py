import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

def carregar_modelo_clip():
    """
    Carrega modelo CLIP pré-treinado
    """
    print("Carregando modelo CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Verificar se GPU disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Modelo carregado em: {device}")
    return model, processor, device

def definir_categorias_cores():
    """
    Define categorias de cores para classificação
    """
    categorias = [
        "drawing with predominantly blue colors",
        "drawing with predominantly red colors", 
        "drawing with predominantly green colors",
        "drawing with predominantly yellow colors",
        "drawing with predominantly purple colors",
        "drawing with predominantly orange colors",
        "colorful drawing with many colors",
        "drawing with mostly black and gray colors",
        "drawing with mostly brown and earth tones",
        "drawing with pastel colors"
    ]
    return categorias

def classificar_imagem_clip(imagem_path, model, processor, device, categorias):
    """
    Classifica uma imagem usando CLIP
    """
    # Carregar e preparar imagem
    imagem = Image.open(imagem_path).convert("RGB")
    
    # Preparar inputs
    inputs = processor(
        text=categorias, 
        images=imagem, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Fazer predição
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Converter para numpy
    probabilidades = probs.cpu().numpy()[0]
    
    # Criar resultado
    resultado = []
    for i, categoria in enumerate(categorias):
        resultado.append({
            'categoria': categoria,
            'probabilidade': round(probabilidades[i] * 100, 2)
        })
    
    # Ordenar por probabilidade
    resultado = sorted(resultado, key=lambda x: x['probabilidade'], reverse=True)
    
    return resultado

def analisar_dataset_clip(caminho_pasta):
    """
    Analisa todas as imagens do dataset com CLIP
    """
    # Carregar modelo
    model, processor, device = carregar_modelo_clip()
    categorias = definir_categorias_cores()
    
    pasta = Path(caminho_pasta)
    resultados_completos = []
    
    print(f"\n🎨 Analisando {len(list(pasta.glob('*.TIF')))} imagens com CLIP...")
    
    for i, arquivo in enumerate(pasta.glob("*.TIF"), 1):
        
        try:
            # Classificar com CLIP
            resultado = classificar_imagem_clip(arquivo, model, processor, device, categorias)
            
            # Salvar resultado completo
            resultado_imagem = {
                'nome': arquivo.name,
                'categoria_principal': resultado[0]['categoria'],
                'confianca_principal': resultado[0]['probabilidade'],
                'categoria_2': resultado[1]['categoria'],
                'confianca_2': resultado[1]['probabilidade'],
                'categoria_3': resultado[2]['categoria'],
                'confianca_3': resultado[2]['probabilidade']
            }
            
            resultados_completos.append(resultado_imagem)
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        print()
    
    return pd.DataFrame(resultados_completos)


if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Analisar com CLIP
    df_clip = analisar_dataset_clip(caminho)
    
    
    # Salvar resultados
    df_clip.to_csv('classificacao_clip_cores.csv', index=False)
    print(f"\n💾 Resultados salvos: classificacao_clip_cores.csv")