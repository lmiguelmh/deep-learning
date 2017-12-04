### HyperFace usando Squeeze & VGG16

- El paper original usa AlexNet.
- No existe una implementación de referencia. Existen implementaciones de terceros incompletas:
	- https://github.com/takiyu/hyperface
	- https://github.com/sourabhvora/HyperFace-with-SqueezeNet
- El presente se encuentra compuesto por 3 notebooks principales:
	- `aflw-dataset.ipynb`
		- Contiene la generación de los conjuntos de entrenamiento y prueba a partir de AFLW
		- Se generan regiones positivas y negativas luego de aplicar Selective Search y Non-Maximum Supression.
	- `training-hyperface-with-squeezenet.ipynb`
		- Contiene un resumen breve del paper original.
		- Usa como red principal SqueezeNet, caracterizada por un tamaño reducido (700K parámetros)
		- Contiene el proceso de construcción del modelo, definición de los mecanismos de pérdida, generación de iterables para los conjuntos de entrenamiento y validación, entrenamiento, y carga y uso del modelo.
		- El modelo final tiene 5,8M de parámetros.
	- `training-hyperface-with-vgg.ipynb`
		- Usa como red principal VGG16.
		- El modelo final tiene 21,6M de parámetros.
