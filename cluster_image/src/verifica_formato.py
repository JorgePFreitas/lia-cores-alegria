import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

def criar_dataset_imagens(caminho_pasta):
    """
    Cria dataset simples com informações das imagens
    """
    pasta = Path(caminho_pasta)
    dados = []
    
    for arquivo in pasta.glob("*.TIF"):
        try:
            # Carregar
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Dados básicos
            altura, largura = img.shape[:2]
            canais = img.shape[2] if len(img.shape) == 3 else 1
            
            # Contagem de cores únicas
            if canais == 3:
                cores_unicas = len(np.unique(img.reshape(-1, 3), axis=0))
                # Pixels brancos (>240 em todos os canais)
                pixels_brancos = np.sum((img[:,:,0] > 240) & (img[:,:,1] > 240) & (img[:,:,2] > 240))
            else:
                cores_unicas = len(np.unique(img))
                pixels_brancos = np.sum(img > 240)
            
            percentual_branco = (pixels_brancos / (largura * altura)) * 100
            
            # Adicionar ao dataset
            dados.append({
                'nome': arquivo.name,
                'largura': largura,
                'altura': altura,
                'canais': canais,
                'cores_unicas': cores_unicas,
                'percentual_branco': round(percentual_branco, 1),
                'tamanho_mb': round(arquivo.stat().st_size / (1024*1024), 2),
                'caminho': str(arquivo)
            })
            
        except Exception as e:
            print(f"❌ Erro com {arquivo.name}: {e}")
    
    # Converter para DataFrame
    df = pd.DataFrame(dados)
    print(f"✅ Dataset criado: {len(df)} imagens")
    
    return df

def salvar_dataset(df, arquivo='dataset_imagens.csv'):
    """
    Salva dataset em CSV
    """
    df.to_csv(arquivo, index=False)
    print(f"💾 Dataset salvo: {arquivo}")

if __name__ == "__main__":
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    
    # Criar dataset
    dataset = criar_dataset_imagens(caminho)
    
    # # Mostrar resumo
    # print(f"\n📊 RESUMO:")
    # print(f"Total: {len(dataset)} imagens")
    # print(f"Dimensões: {dataset['largura'].iloc[0]}x{dataset['altura'].iloc[0]}")
    # print(f"Cores médias: {dataset['cores_unicas'].mean():.0f}")
    # print(f"Fundo branco médio: {dataset['percentual_branco'].mean():.1f}%")
    
    # Salvar
    salvar_dataset(dataset)