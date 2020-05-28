package omikuji

// #cgo LDFLAGS: -L${SRCDIR}/lib -L${SRCDIR}/c-api/target/release/ -lomikuji
// #cgo CFLAGS: -g -Wall -I${SRCDIR}/lib
// #include "omikuji.h"
import "C"

import (
	"fmt"
	"log"
	"unsafe"
)

const NUMTHREADS = 2

// A model object. Effectively a wrapper around pointers to the
// omikuji model and threadpool structs
type Model struct {
	path   string
	handle *C.OMIKUJI_Model
	pool   *C.OMIKUJI_ThreadPool
}

// Opens a model from a path and returns a model object
func Open(path string) *Model {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	model := Model{
		path:   path,
		handle: C.load_omikuji_model(cpath),
		pool:   C.init_omikuji_thread_pool(NUMTHREADS),
	}
	fmt.Printf("model: %#v\n", model)
	return &model
}

// Closes a model handle
func (model *Model) Close() error {
	if model == nil {
		return nil
	}

	var cModel *C.OMIKUJI_Model = model.handle
	defer C.free(unsafe.Pointer(cModel))

	var cPool *C.OMIKUJI_ThreadPool = model.pool
	defer C.free(unsafe.Pointer(cPool))

	C.free_omikuji_model(cModel)
	C.free_omikuji_thread_pool(cPool)
	return nil
}

// Performs model prediction
func (model *Model) PredictDefault(keys []uint32, vals []float32) ([]uint32, []float32) {
	labels, scores := model.Predict(keys, vals, 10, 10)
	return labels, scores
}

func (model *Model) Predict(keys []uint32, vals []float32, beamSize int, topK int) ([]uint32, []float32) {
	if model == nil {
		log.Fatalln("model not initialized; aborting")
	}

	inputLen := len(keys)
	if len(vals) != inputLen {
		log.Fatalln("keys and values have different length; aborting")
	}

	outputLabels := make([]uint32, topK)
	outputScores := make([]float32, topK)

	var cModel *C.OMIKUJI_Model = model.handle
	defer C.free(unsafe.Pointer(cModel))

	var cPool *C.OMIKUJI_ThreadPool = model.pool
	defer C.free(unsafe.Pointer(cPool))

	r := C.omikuji_predict(
		cModel,
		C.size_t(beamSize),              // beam size
		C.size_t(inputLen),              // length of keys and vals input arrays
		(*C.uint32_t)(&keys[0]),         // *feature_indices
		(*C.float)(&vals[0]),            // const float *feature_values,
		C.size_t(topK),                  // output length
		(*C.uint32_t)(&outputLabels[0]), // uint32_t *output_labels,
		(*C.float)(&outputScores[0]),    // float *output_scores,
		cPool,                           // const OMIKUJI_ThreadPool *thread_pool_ptr
	)
	return outputLabels[0:r], outputScores[0:r]
}

/*
func (handle *Model) OriginalPredict(query string) (Predictions, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	// Call the Predict function defined in cbits.cpp
	// passing in the model handle and the query string
	r := C.Predict(handle.handle, cquery, 3)
	// the C code returns a c string which we need to
	// convert to a go string
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	// unmarshal the json results into the predictions
	// object. See https://blog.golang.org/json-and-go
	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}

	return predictions, nil
}

func (handle *Model) Analogy(query string) (Analogs, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	r := C.Analogy(handle.handle, cquery)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	analogies := []Analog{}
	err := json.Unmarshal([]byte(js), &analogies)
	if err != nil {
		return nil, err
	}

	return analogies, nil
}

*/
