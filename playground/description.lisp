;; network numberrecognizer ++layers=1 ++inputsize=64 ++outputsize=10
;; layer numberrecognizer 0
;;   ++neurons=10
;;   ++activation=softmax

;; (network "number-recognizer"
;;          (input-size 64)
;;          (layers
;;            (layer 0
;;                   (type "fully-connected")
;;                   (neurons 16)
;;                   (activation "relu"))
;;            (layer 1
;;                   (type "fully-connected")
;;                   (neurons 10)
;;                   (activation "softmax"))))
;;
;; (train
;;   (dataset "dataset.txt"
;;            (loader "dot-and-X"))
;;   (epochs 5000)
;;   (learning-rate 0.01))

;; (network "number-recognizer"
;;   (input-size 64)
;;   (layers
;;     (layer (id 0) (type "fully-connected") (neurons 16) (activation "relu"))
;;     (layer (id 1) (type "fully-connected") (neurons 10) (activation "softmax")))
;;   (train
;;     (dataset "dataset.txt")
;;     (epochs 5000)
;;     (learning-rate 0.02)))

(define "number-recognizer"
             :input-size 64
             (layers
               (layer :neurons 16 :activation "relu")
               (layer :neurons 10 :activation "softmax")))


(train "number-recognizer"
       :data "data.txt"
       :epochs 20000
       :learning-rate 0.05)

(test "number-recognizer"
      :data "test.txt")
