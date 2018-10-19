(define (problem prob)
    (:domain puzzle)

    (:objects n0 n1 n2 n3 n4 n5 n6 n7 n8 -num 
            l0 l1 l2 l3 l4 l5 l6 l7 l8 -loc)

    (:init (At n1 l0) (At n2 l1) (At n3 l2) (At n7 l3) 
            (At n8 l4) (At n0 l5) (At n6 l6) (At n4 l7) 
            (At n5 l8) (Next l0, l1) (Next l1 l0) (Next l0 l3) 
            (Next l3 l0) (Next l1 l2) (Next l2 l1) (Next l1 l4) 
            (Next l4 l1) (Next l2 l5) (Next l5 l2) (Next l3 l4) 
            (Next l4 l3) (Next l3 l6) (Next l6 l3) (Next l4 l5)
            (Next l5 l4) (Next l4 l7) (Next l7 l4) (Next l5 l8) 
            (Next l8 l5) (Next l6 l7) (Next l7 l6) (Next l7 l8) 
            (Next l8 l7)
    )

    (:goal (and (At n1 l0) (At n2 l1) (At n3 l2) (At n4 l3) 
                (At n5 l4) (At n6 l5) (At n7 l6) (At n8 l7) 
                (At n0 l8)
            )
    )
)