import os, cv2
import numpy as np

#funzione per applicare preprocessing ad una singola immagine
def img_preprocess(crop, detector, read_path):
    #carica l'immagine dalla cartella specifica del soggetto
    image = cv2.imread(read_path)

    #se c'è un problema con l'immagine evita errori successivi
    if image is None:
        return None

    #porta in scala di grigio in previsione dell'applicazione del local binary pattern
    gray_scale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #se l'immagine è reale deve essere croppata
    if crop:
        #rileva il volto
        faces = detector.detectMultiScale(gray_scale_img, scaleFactor=1.1, minNeighbors=9)
        
        #se è rilevato più di un volto l'immagine è scartata
        if len(faces) == 1:
 
            face = faces[0]
            #trasla per far concidere il centro del volto con il centro dell'immagine
            width, height = face[2], face[3]
            img_center = (gray_scale_img.shape[1] // 2, gray_scale_img.shape[0] // 2)
            face_center = (face[0] + width // 2, face[1] + height // 2)

            #calcolo i valori della matrice di traslazione
            tx = img_center[0] - face_center[0]
            ty = img_center[1] - face_center[1]

            #applico la trasformazione
            M_affine = np.array([[1, 0, tx], [0, 1, ty]]).astype(np.float32)
            gray_scale_img = cv2.warpAffine(gray_scale_img, M_affine, dsize=(gray_scale_img.shape[1], gray_scale_img.shape[0]))
         
            #scala in base all'altezza del volto individuata (90% dell'altezza dell'immagine)
            target_height = 0.90 * gray_scale_img.shape[0]
            scale = target_height / height

            #applica la scalatura al centro
            M_affine = cv2.getRotationMatrix2D(img_center, 0, scale)
            gray_scale_img = cv2.warpAffine(gray_scale_img, M_affine, dsize=(gray_scale_img.shape[1], gray_scale_img.shape[0]))

        else:
            return None
    
    return gray_scale_img
    
def main():
    #recupera le liste dei nomi di tutti i soggetti (cartelle) real e fake
    real_subjects = [f for f in os.listdir(os.path.join("raw_dataset", "realfaces"))]
    fake_subjects = [f for f in os.listdir(os.path.join("raw_dataset", "fakefaces"))]

    #crea la cartella in cui salvare le immagini_preprocessate, se non esiste
    os.makedirs("processed_dataset", exist_ok=True)

    #inizializza il Cascade solo una volta
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    
    #preprocessing e cropping delle immagini reali per ogni soggetto
    for i, real in enumerate(real_subjects):
        
        #seleziono lo specifico soggetto/cartella da cui prelevare le immagini
        read_path = os.path.join("raw_dataset", "realfaces", real)

        #Le immagini di un soggetto vengono salvate in una apposita cartella "real_n" con n il numero del soggetto
        save_path = os.path.join("processed_dataset", f"real_{i}")

        #se la cartella per salvare le immagini del soggetto non esiste, viene creata
        if not(os.path.exists(save_path)):
            os.mkdir(save_path)

        #elabora e salva ogni immagine del soggetto
        for j, img in enumerate(os.listdir(read_path)):
           
            processed_img = img_preprocess(True, face_detector, os.path.join(read_path, img))

            #salva l'immagine solo se non è stata scartata
            if processed_img is not None:

                save_path_j = os.path.join(save_path, f"{j}.jpg")

                cv2.imwrite(save_path_j, processed_img)
    
    #preprocessing e cropping delle immagini fake per ogni soggetto
    for i, fake in enumerate(fake_subjects):

         #seleziono lo specifico soggetto/cartella da cui prelevare le immagini
        read_path = os.path.join("raw_dataset", "fakefaces", fake)

        #Le immagini di un soggetto vengono salvate in una apposita cartella "fake_n" con n il numero del soggetto
        save_path = os.path.join("processed_dataset", f"fake_{i}")
        

        if not(os.path.exists(save_path)):
            os.mkdir(save_path)

        for j, img in enumerate(os.listdir(read_path)):

            save_path_j = os.path.join(save_path, f"{j}.jpg")

            processed_img = img_preprocess(False, face_detector, os.path.join(read_path, img))
            #in questo caso non è necessario il controllo precendente poiché le immagini fake non vengono croppate
            cv2.imwrite(save_path_j, processed_img)


if __name__ == "__main__":
    main()