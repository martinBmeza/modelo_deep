Este esquema es tomado de la pagina web: 
https://deeps.site/blog/2019/12/07/dl-project-structure/

Inicialmente esta pensado para proyectos que tengan relacion con el procesamiento de audio, pero es extrapolable a cualquier situacion que implique implementar un modelo de redes neuronales, entrenarlo con un set de datos preprocesados, y analizar su funcionamiento y resultados. 

Es importante setear correctamente la variable MAIN PATH en los archivos: 

-build_dataset.py
-data_loader.py
-
-

MAIN PATH tiene que ser la ruta a la carpeta principal donde se guardan todos los archivos del proyecto. esto permite independizarse de la plataforma en la que se trabaje siempre que se mantenga esta estructura. 

-dell notebook = '/home/mrtn/Documents/TESIS/de-reverb/Source/MODELOS/modelo_prueba'

-Prueba para funcionar en PC con CUDA 
