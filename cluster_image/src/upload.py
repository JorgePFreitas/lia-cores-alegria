import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def carregar_imagens(caminho_pasta):
    """
    Carrega imagens de forma simples - OpenCV + PIL backup
    """
    pasta = Path(caminho_pasta)
    imagens = []
    nomes = []
    
    print(f"📁 Carregando de: {pasta.name}")
    
    for arquivo in pasta.glob("*.TIF"):  # Apenas TIF por enquanto
        # Tentar PIL (melhor para TIF)
        try:
            pil_img = Image.open(arquivo)
            img_array = np.array(pil_img)
            
            # Converter RGB para BGR se necessário
            if len(img_array.shape) == 3 and pil_img.mode == 'RGB':
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            imagens.append(img)
            nomes.append(arquivo.name)
            print(f"✅ {arquivo.name} - {img.shape}")
            
        except Exception as e:
            print(f"❌ {arquivo.name}: {e}")
    
    print(f"\n📊 Total: {len(imagens)} imagens")
    return imagens, nomes

def ver_imagem(imagens, nomes, indice=0):
    """
    Mostra uma imagem específica
    """
    if indice < len(imagens):
        cv2.imshow(f'{nomes[indice]}', imagens[indice])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def info_imagem(img, nome):
    """
    Informações básicas da imagem
    """
    altura, largura = img.shape[:2]
    canais = img.shape[2] if len(img.shape) == 3 else 1
    
    print(f"\n🖼️  {nome}:")
    print(f"   Tamanho: {largura} x {altura}")
    print(f"   Canais: {canais}")
    print(f"   Tipo: {img.dtype}")
    
    return {
        'largura': largura,
        'altura': altura,
        'canais': canais,
        'tipo': str(img.dtype)
    }

if __name__ == "__main__":
    # Carregar
    caminho = r"C:\Users\jorge\Desktop\Projetos\Lia²\lia-cores-alegria\cluster_image\CA_processada"
    imagens, nomes = carregar_imagens(caminho)
    
    # Info da primeira
    if imagens:
        info_primeira = info_imagem(imagens[0], nomes[0])
        
        # Ver primeira imagem?
        ver = input("\n🔍 Ver primeira imagem? (s/n): ")
        if ver.lower() in ['s', 'sim']:
            ver_imagem(imagens, nomes, 0)