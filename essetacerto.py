import cv2
import numpy as np
import time
import requests
import simplejson
import math

#####################################################
############### LEGENDA DE CORES ####################
#####################################################
# Deve-se mudar o "upper" e o "lower" de cada cor principal e secundárias dos robôs

##### VERMELHO #####
# Upper: [10, 255, 255] ou [160, 100, 100] a [179, 255, 255]
# Lower: [0, 100, 100]

##### O ROXO PODE VARIAR SENDO MAIS PRÓXIMO DO AZUL OU VERMELHO #####
# Para tons mais próximos ao azul: Lower: [110, 100, 100], Upper: [130, 255, 255]
# Para tons mais próximos ao vermelho: Lower: [140, 100, 100], Upper: [170, 255, 255]

##### CIANO #####
# Lower: [85, 100, 100]
# Upper: [100, 255, 255]

##### MAGENTA #####
# Lower: [140, 100, 100]
# Upper: [170, 255, 255]

##### AZUL #####
# Lower: [90, 100, 100]
# Upper: [130, 255, 255]

##### VERDE #####
# Lower: [30, 40, 20]
# Upper: [90, 255, 255]

##### AMARELO #####
# Lower: [20, 100, 100]
# Upper: [30, 255, 255]

##### ROSA #####
# Lower: [140, 50, 50]
# Upper: [170, 255, 255]


def enhance_contrast(image):
    # Aumenta o contraste da imagem
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel_size = 5  # Tamanho do kernel para a suavização
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return enhanced_image

def detect_colors_with_contours():
    #######################
    #esp32_central_url1 = "http://200.128.142.193/cood(y):retanguloR1"  # URL para a coordenada do "y" do retangulo robo 1 
    #esp32_central_url2 = "http://200.128.142.193/cood(x):retanguloR1"  # URL para a coordenada do "x" do retangulo robo 1 
    #esp32_central_url3 = "http://200.128.142.193/cood(y):circuloR1"  # URL para a coordenada do "y" do bola robo 1 
    #esp32_central_url4 = "http://200.128.142.193/cood(x):circuloR1"  # URL para a coordenada do "x" do bola robo 1 

    #esp32_central_url5 = "http://200.128.142.193/cood(y):retanguloR2"  # URL para a coordenada do "y" do retangulo robo 2 
    #esp32_central_url6 = "http://200.128.142.193/cood(x):retanguloR2"  # URL para a coordenada do "x" do retangulo robo 2 
    #esp32_central_url7 = "http://200.128.142.193/cood(y):circuloR2"  # URL para a coordenada do "y" do bola robo 2 
    #esp32_central_url8 = "http://200.128.142.193/cood(x):circuloR2"  # URL para a coordenada do "x" do bola robo 2 

    #esp32_central_url9 = "http://200.128.142.193/cood(y):retanguloR3"  # URL para a coordenada do "y" do retangulo robo 3 
    #esp32_central_url10 = "http://200.128.142.193/cood(x):retanguloR3"  # URL para a coordenada do "x" do retangulo robo 3 
    #esp32_central_url11 = "http://200.128.142.193/cood(y):circuloR3"  # URL para a coordenada do "y" do bola robo 3 
    #esp32_central_url12 = "http://200.128.142.193/cood(x):circuloR3"  # URL para a coordenada do "x" do bola robo 3 

    #esp32_central_url13 = "http://200.128.142.193/cood(y):retanguloR4"  # URL para a coordenada do "y" do retangulo robo 4 
    #esp32_central_url14 = "http://200.128.142.193/cood(x):retanguloR4"  # URL para a coordenada do "x" do retangulo robo 4 
    #esp32_central_url15 = "http://200.128.142.193/cood(y):circuloR4"  # URL para a coordenada do "y" do bola robo 4 
    #esp32_central_url16 = "http://200.128.142.193/cood(x):circuloR4"  # URL para a coordenada do "x" do bola robo 4 
    

    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    display_interval = 1.0  # Intervalo de exibição em segundos
    last_display_time = 0

    while True:
        ret, frame = cap.read()

        if ret:
            enhanced_frame = enhance_contrast(frame)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detecção de objetos azuis com círculo rosa ou amarelo

            # Mudar a cor principal do robô (Upper e Lower)
            lower_main = np.array([90, 100, 100])
            upper_main = np.array([130, 255, 255])
            main_mask = cv2.inRange(hsv_frame, lower_main, upper_main)
            main_contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in main_contours:
                if cv2.contourArea(contour) > 100:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Detecção de círculo rosa ou amarelo dentro do objeto azul
                        roi = frame[max(0, cY - 15):min(frame.shape[0], cY + 15),
                                    max(0, cX - 15):min(frame.shape[1], cX + 15)]

                        if roi.size == 0:
                            continue

                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        # Mudar o Upper e o Lower da cor do círculo (cor secundária 1)
                        lower_secondary1 = np.array([140, 50, 50])
                        upper_secondary1 = np.array([170, 255, 255])
                        secondary1_mask = cv2.inRange(hsv_roi, lower_secondary1, upper_secondary1)
                        secondary1_contours, _ = cv2.findContours(secondary1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Mudar o Upper e o Lower da cor do círculo (cor secundária 2)
                        lower_secondary2 = np.array([20, 100, 100])
                        upper_secondary2 = np.array([30, 255, 255])
                        secondary2_mask = cv2.inRange(hsv_roi, lower_secondary2, upper_secondary2)
                        secondary2_contours, _ = cv2.findContours(secondary2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for secondary1_contour in secondary1_contours:
                            secondary1_moment = cv2.moments(secondary1_contour)
                            if secondary1_moment["m00"] != 0:
                                secondary1_cX = int(secondary1_moment["m10"] / secondary1_moment["m00"])
                                secondary1_cY = int(secondary1_moment["m01"] / secondary1_moment["m00"])

                                secondary1_cX += max(0, cX - 15)
                                secondary1_cY += max(0, cY - 15)

                                # Objeto azul com círculo rosa
                                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                                cv2.drawContours(frame, [secondary1_contour], -1, (0, 0, 255), 2)
                                print(f"Objeto azul com círculo rosa: Objeto: ({cX}, {cY}), Círculo rosa: ({secondary1_cX}, {secondary1_cY})")

                                #data = {"cood(y):retanguloR1": cY }  # Dados para a coordenada "y" do retangulo do robo 1
                                #response = requests.post(esp32_central_url1, json=data)

                                #data = {"cood(x):retanguloR1": cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url2, json=data)

                                #data = {"cood(y):circuloR1": secondary1_cY}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url3, json=data)

                                #data = {"cood(x):circuloR1": secondary1_cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url4, json=data)

                                #####################################################
                                #Cálculo do arctg de orientação da reta

                                x_inicio = cX
                                y_inicio = cY
                                x_fim = secondary1_cX
                                y_fim = secondary1_cY

                                # Calculando as diferenças entre as coordenadas para obter o vetor direção
                                delta_x = x_fim - x_inicio
                                delta_y = y_fim - y_inicio

                                # Calculando o ângulo em radianos usando a função atan2
                                angulo_radianos = math.atan2(delta_y, delta_x)

                                # Convertendo o ângulo para graus
                                angulo_graus = math.degrees(angulo_radianos)

                                # Calculando a tangente do ângulo
                                tangente_angulo = math.tan(angulo_radianos)

                                print (f"Angulo do segmento de reta do robô azul com círculo rosa: {angulo_graus}")
                                print (f"A tangente desse ângulo é: {tangente_angulo}")


                        for secondary2_contour in secondary2_contours:
                            secondary2_moment = cv2.moments(secondary2_contour)
                            if secondary2_moment["m00"] != 0:
                                secondary2_cX = int(secondary2_moment["m10"] / secondary2_moment["m00"])
                                secondary2_cY = int(secondary2_moment["m01"] / secondary2_moment["m00"])

                                secondary2_cX += max(0, cX - 15)
                                secondary2_cY += max(0, cY - 15)

                                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                                cv2.drawContours(frame, [secondary2_contour], -1, (0, 255, 255), 2)
                                print(f"Objeto azul com círculo amarelo: Objeto: ({cX}, {cY}), Círculo amarelo: ({secondary2_cX}, {secondary2_cY})")

                                #data = {"cood(y):retanguloR2": cY }  # Dados para a coordenada "y" do retangulo do robo 2
                                #response = requests.post(esp32_central_url5, json=data)

                                #data = {"cood(x):retanguloR2": cX}  # Dados para a coordenada "x" do retangulo do robo 2
                                #response = requests.post(esp32_central_url6, json=data)

                                #data = {"cood(y):circuloR2": secondary2_cY}  # Dados para a coordenada "x" do retangulo do robo 2
                                #response = requests.post(esp32_central_url7, json=data)

                                #data = {"cood(x):circuloR2": secondary2_cX}  # Dados para a coordenada "x" do retangulo do robo 2
                                #response = requests.post(esp32_central_url8, json=data)

                                #####################################################
                                #Cálculo do arctg de orientação da reta

                                x_inicio = cX
                                y_inicio = cY
                                x_fim = secondary2_cX
                                y_fim = secondary2_cY

                                # Calculando as diferenças entre as coordenadas para obter o vetor direção
                                delta_x = x_fim - x_inicio
                                delta_y = y_fim - y_inicio

                                # Calculando o ângulo em radianos usando a função atan2
                                angulo_radianos = math.atan2(delta_y, delta_x)

                                # Convertendo o ângulo para graus
                                angulo_graus = math.degrees(angulo_radianos)

                                # Calculando a tangente do ângulo
                                tangente_angulo = math.tan(angulo_radianos)

                                print (f"Angulo do segmento de reta do robô azul com círculo amarelo: {angulo_graus}")
                                print (f"A tangente desse ângulo é: {tangente_angulo}")

            # Detecção de objetos verdes com círculo rosa ou amarelo

            # Mudar a cor principal do robô (Upper e Lower)
            lower_main2 = np.array([30, 40, 20])
            upper_main2 = np.array([90, 255, 255])
            main2_mask = cv2.inRange(hsv_frame, lower_main2, upper_main2)
            main2_contours, _ = cv2.findContours(main2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in main2_contours:
                if cv2.contourArea(contour) > 100:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Detecção de círculo rosa ou amarelo dentro do objeto verde
                        roi = frame[max(0, cY - 15):min(enhanced_frame.shape[0], cY + 15),
                                    max(0, cX - 15):min(enhanced_frame.shape[1], cX + 15)]

                        if roi.size == 0:
                            continue

                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        # Mudar o Upper e o Lower da cor do círculo (cor secundária 1)
                        lower_secondary1 = np.array([140, 50, 50])
                        upper_secondary1 = np.array([170, 255, 255])
                        secondary1_mask = cv2.inRange(hsv_roi, lower_secondary1, upper_secondary1)
                        secondary1_contours, _ = cv2.findContours(secondary1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Mudar o Upper e o Lower da cor do círculo (cor secundária 2)
                        lower_secondary2 = np.array([20, 100, 100])
                        upper_secondary2 = np.array([30, 255, 255])
                        secondary2_mask = cv2.inRange(hsv_roi, lower_secondary2, upper_secondary2)
                        secondary2_contours, _ = cv2.findContours(secondary2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for secondary1_contour in secondary1_contours:
                            secondary1_moment = cv2.moments(secondary1_contour)
                            if secondary1_moment["m00"] != 0:
                                secondary1_cX = int(secondary1_moment["m10"] / secondary1_moment["m00"])
                                secondary1_cY = int(secondary1_moment["m01"] / secondary1_moment["m00"])

                                # Coordenadas do centro do círculo em relação à imagem inteira
                                secondary1_cX += max(0, cX - 15)
                                secondary1_cY += max(0, cY - 15)

                                # Objeto verde com círculo rosa
                                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                                cv2.drawContours(frame, [secondary1_contour], -1, (0, 0, 255), 2)
                                print(f"Objeto verde com círculo rosa: Objeto: ({cX}, {cY}), Círculo rosa: ({secondary1_cX}, {secondary1_cY})")

                                #data = {"cood(y):retanguloR3": cY }  # Dados para a coordenada "y" do retangulo do robo 1
                                #response = requests.post(esp32_central_url9, json=data)

                                #data = {"cood(x):retanguloR3": cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url10, json=data)

                                #data = {"cood(y):circuloR3": secondary1_cY}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url11, json=data)

                                #data = {"cood(x):circuloR3": secondary1_cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url12, json=data)

                                #####################################################
                                #Cálculo do arctg de orientação da reta

                                x_inicio = cX
                                y_inicio = cY
                                x_fim = secondary1_cX
                                y_fim = secondary1_cY

                                # Calculando as diferenças entre as coordenadas para obter o vetor direção
                                delta_x = x_fim - x_inicio
                                delta_y = y_fim - y_inicio

                                # Calculando o ângulo em radianos usando a função atan2
                                angulo_radianos = math.atan2(delta_y, delta_x)

                                # Convertendo o ângulo para graus
                                angulo_graus = math.degrees(angulo_radianos)

                                # Calculando a tangente do ângulo
                                tangente_angulo = math.tan(angulo_radianos)

                                print (f"Angulo do segmento de reta do robô verde com círculo rosa: {angulo_graus}")
                                print (f"A tangente desse ângulo é: {tangente_angulo}")

                        for secondary2_contour in secondary2_contours:
                            secondary2_moment = cv2.moments(secondary2_contour)
                            if secondary2_moment["m00"] != 0:
                                secondary2_cX = int(secondary2_moment["m10"] / secondary2_moment["m00"])
                                secondary2_cY = int(secondary2_moment["m01"] / secondary2_moment["m00"])

                                # Coordenadas do centro do círculo em relação à imagem inteira
                                secondary2_cX += max(0, cX - 15)
                                secondary2_cY += max(0, cY - 15)

                                # Objeto verde com círculo amarelo
                                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                                cv2.drawContours(frame, [secondary2_contour], -1, (0, 255, 255), 2)
                                print(f"Objeto verde com círculo amarelo: Objeto: ({cX}, {cY}), Círculo amarelo: ({secondary2_cX}, {secondary2_cY})")

                                #data = {"cood(y):retanguloR4": cY }  # Dados para a coordenada "y" do retangulo do robo 1
                                #response = requests.post(esp32_central_url13, json=data)

                                #data = {"cood(x):retanguloR4": cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url14, json=data)

                                #data = {"cood(y):circuloR4": secondary2_cY}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url15, json=data)

                                #data = {"cood(x):circuloR4": secondary2_cX}  # Dados para a coordenada "x" do retangulo do robo 1
                                #response = requests.post(esp32_central_url16, json=data)

                                #####################################################
                                #Cálculo do arctg de orientação da reta

                                x_inicio = cX
                                y_inicio = cY
                                x_fim = secondary2_cX
                                y_fim = secondary2_cY

                                # Calculando as diferenças entre as coordenadas para obter o vetor direção
                                delta_x = x_fim - x_inicio
                                delta_y = y_fim - y_inicio

                                # Calculando o ângulo em radianos usando a função atan2
                                angulo_radianos = math.atan2(delta_y, delta_x)

                                # Convertendo o ângulo para graus
                                angulo_graus = math.degrees(angulo_radianos)

                                # Calculando a tangente do ângulo
                                tangente_angulo = math.tan(angulo_radianos)

                                print (f"Angulo do segmento de reta do robô verde com círculo amarelo: {angulo_graus}")
                                print (f"A tangente desse ângulo é: {tangente_angulo}")
                        
            # Detecção de círculos laranja isolados (Bola)
            lower_orange = np.array([0, 100, 100])
            upper_orange = np.array([20, 255, 255])
            orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in orange_contours:
                if cv2.contourArea(contour) > 100:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Coordenadas do centro do círculo em relação à imagem inteira
                        cv2.drawContours(frame, [contour], -1, (0, 165, 255), 2)  # Contorno laranja
                        print(f"Círculo laranja isolado: ({cX}, {cY})")
        

            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                last_display_time = current_time

            cv2.imshow("Detecção de Cores Azul, Rosa, Verde e Amarelo", frame)
            time.sleep(display_interval)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_colors_with_contours()
