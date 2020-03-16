WORK=/data2/hoaphan/onnxruntime/
SDK_ONNX=/data2/hoaphan/onnxruntime/sdk/sysroots/

onnxruntime: tiny_yolo_v3.cpp
	${CXX} -std=c++14 tiny_yolo_v3.cpp box.cpp image.cpp \
	-DONNX_ML \
	-I /tftpboot/onnxruntime/include/onnxruntime/core/session/ \
	-I /tftpboot/stb/ \
	-L /tftpboot/onnxruntime/build/Linux/RelWithDebInfo/ \
	-L /tftpboot/onnxruntime/build/Linux/RelWithDebInfo/external/re2/ \
	-L /tftpboot/onnxruntime/build/Linux/RelWithDebInfo/external/protobuf/cmake/ \
	-L /tftpboot/onnxruntime/build/Linux/RelWithDebInfo/onnx/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
	-lautoml_featurizers \
	-lonnxruntime_framework \
	-lonnxruntime_optimizer \
	-lonnxruntime_graph \
	-lonnxruntime_common \
	-lonnx_proto \
	-lprotobuf \
	-lre2 \
	-lonnxruntime_util \
	-lonnxruntime_mlas \
	-lonnx \
	-ljpeg -ltbb -ltiff -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o tiny_yolo_v3

clean:
	rm -rf *.o tiny_yolo_v3
