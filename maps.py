m = {
    "default": {
        "shape": [10,10],
        "max_speed": .25,
        "boxes": {
            # "everywhere": {
            #     "coords": [[0,0],[10,10]], 
            #     "reward": 0,
            # },
            "init": {
                "coords": [[1,1],[4,9]],
                "init_weight": 1,
                "edge_colour": "b"
            },
            "goal": {
                "coords": [[6,6],[8,8]],
                "reward": 100,
                "continuation_prob": 0,
                "face_colour": "g"
            },
        },
        "walls": {
            "barrier": {
                "coords": [[5,3],[5,7]],
            }
        },
        "point_attractors": {
            "dist_to_goal": {
                "coords": [7,7],
                "reward": -0.1,
                "colour": "w"
            }
        },
        "line_attractors": {
            "dist_to_line": {
                "coords": [[0,0],[10,10]],
                "reward": -0.1,
                "colour": "b"
            }
        }
    },
    "curiosity": {
        "shape": [10,10],
        "max_speed": .25,
        "curiosity": {
            "num": 10,
            "reward": -1
        },
        "boxes": {
            "everywhere": {
                "coords": [[0,0],[10,10]], 
                "init_weight": 1,
                "trigger": [["everywhere", False]]
            }
        }   
    }
}