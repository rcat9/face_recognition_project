import numpy as np

def convert_color(color, scale="MINUS_ONE_TO_ONE"):
    """色値を-1〜1のスケールに変換"""
    return (color - 127.5) * 0.0078125

def convert_image_to_model_input(image, color_space="RGB"):
    """画像をモデル入力用の配列に変換"""
    height, width, _ = image.shape
    model_input = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            r = convert_color(image[y, x, 0])
            g = convert_color(image[y, x, 1])
            b = convert_color(image[y, x, 2])
            
            if color_space == "RGB":
                model_input[y, x] = [r, g, b]
            elif color_space == "BGR":
                model_input[y, x] = [b, g, r]
    
    return model_input