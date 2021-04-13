import random
import math

SCENARIOS = [
    # [
    #     # Narrative
    #     [
    #         (1, 27, 'Owner Leaves Apartment'),
    #         (2, 27, 'Owner Opens Phone'),
    #         (3, 22, 'Owner Arrives at Coffee Shop'),
    #         (3.5, 17, 'Owner Puts Down Phone'),
    #         (4, 27, 'Owner Sits Down'),
    #         (5, 27, 'Rush of Customers'),
    #         (6, 27, 'Owner Gets up to Order'),
    #         (7.5, 27, 'Attempted Theft')
    #     ],
    #     # Alpha
    #     [
    #         (0, 150),
    #         (1, 150),
    #         (1, 3),
    #         (3, 3),
    #         (3, 50),
    #         (10, 50)
    #     ],
    #     # Privacy
    #     [
    #         (0, 1, .1),
    #         (1, 1, .1),
    #         (1, 25, .2),
    #         (3, 25, .2),
    #         (3, 5, .3),
    #         (5, 5, .3),
    #         (7, 25, .4),
    #         (10, 25, .4)
    #     ],
    #     # Unfamiliarity
    #     [
    #         (0, 0, .1),
    #         (1, 0, .1),
    #         (1, 20, .2),
    #         (3, 20, .2),
    #         (3, 2, .3),
    #         (5, 2, .3),
    #         (7, 20, .4),
    #         (10, 20, .4)
    #     ],
    #     # Proximity
    #     [
    #         (0, .75, .25),
    #         (1, .75, .25),
    #         (1, 1, .25),
    #         (3.5, 1, .25),
    #         (3.5, .7, .25),
    #         (3.75, .7, .25),
    #         (6, .7, .25),
    #         (6.1, .4, .25),
    #         (7.5, .2, .25),
    #         (10, 0, .25)
    #     ]
    # ],
    # [
    #     # narrative
    #     [
    #         (1, 27, 'Owner Boards Train'),
    #         (3, 27, 'Train Stops for Passengers'),
    #         (5, 27, 'Train Stops for Passengers'),
    #         (6, 27, 'Owners Leaves for Bathroom'),
    #         (8, 27, 'Train Stops for Passengers'),
    #         (9, 27, 'Train Arrives Downtown'),
    #         (9.5, 22, 'Owner Disembarks')
    #     ],
    #     # Alpha
    #     [
    #         (0, 3),
    #         (10, 3)
    #     ],
    #     # Privacy
    #     [
    #         (0, 1, .1),
    #         (1, 1, .1),
    #         (3, 1, .1),
    #         (3, 3, .1),
    #         (5, 3, .1),
    #         (5, 6, .1),
    #         (8, 6, .1),
    #         (8, 15, .1),
    #         (9, 15, .1),
    #         (9.5, 20, .1),
    #         (10, 20, .1)
    #     ],
    #     # Unfamiliarity
    #     [
    #         (0, 1, .1),
    #         (1, 1, .1),
    #         (3, 1, .1),
    #         (3, 3, .1),
    #         (5, 3, .1),
    #         (5, 5, .1),
    #         (8, 5, .1),
    #         (8, 12, .1),
    #         (9, 12, .1),
    #         (9.5, 16, .1),
    #         (10, 16, .1)
    #     ],
    #     # Proximity
    #     [
    #         (0, 1, .2),
    #         (2, 1, .2),
    #         (6, 1, .2),
    #         (6.3, 0, .2),
    #         (6.7, 0, .2),
    #         (7, 1, .2),
    #         (10, 1, .2)
    #     ]
    # ],
    # [
    #     # narrative
    #     [
    #         (1, 25, 'Guests Begin to Arrive'),
    #         (7, 25, 'Guests Begin to Depart')
    #     ],
    #     # Alpha
    #     [
    #         (0, 150),
    #         (10, 150)
    #     ],
    #     # Privacy
    #     [
    #         (0, 1, .2),
    #         (1, 1, .2),
    #         (2, 15, .2),
    #         (7, 15, .2),
    #         (9, 1, .2),
    #         (10, 1, .2)
    #     ],
    #     # Unfamiliarity
    #     [
    #         (0, 0, .2),
    #         (1, 0, .2),
    #         (2, 6, .2),
    #         (8, 6, .2),
    #         (9, 0, .2),
    #         (10, 0, .2)
    #     ],
    #     # Proximity
    #     [(i/4, (random.random() * 0.5) + 0.5 if i > 4 and i < 36 else 1, (random.random() * 0.7) + 0.1)
    #      for i in range(41)]
    # ],
    # [
    #     # narrative
    #     [
    #         (0, 27, 'Owner is Walking Down the Street'),
    #         (2, 27, 'Owner Stops for Coffee'),
    #         (3.75, 27, 'Owner Continues Walking Down the Street'),
    #         (5, 22, 'Owner Arrives at Work'),
    #         (8, 27, 'Owner Leaves Work'),
    #         (9, 22, 'Owner Boards Crowded Bus')
    #     ],
    #     # Alpha
    #     [
    #         (0, 1),
    #         (2, 1),
    #         (2, 25),
    #         (3.75, 25),
    #         (3.75, 3),
    #         (5, 3),
    #         (5, 100),
    #         (8, 100),
    #         (8, 3),
    #         (10, 3)
    #     ],
    #     # Privacy
    #     [
    #         (0, 25, .3),
    #         (2, 25, .4),
    #         (2, 15, .1),
    #         (3.75, 15, .1),
    #         (3.75, 20, .3),
    #         (5, 20, .3),
    #         (5, 15, .2),
    #         (8, 15, .2),
    #         (8, 20, .1),
    #         (9, 20, .1),
    #         (9, 30, .1),
    #         (10, 30, .1)
    #     ],
    #     # Unfamiliarity
    #     [
    #         (0, 16, .1),
    #         (2, 16, .1),
    #         (2, 8, .4),
    #         (3.75, 8, .25),
    #         (3.75, 16, .2),
    #         (5, 16, .2),
    #         (5, 4, .2),
    #         (8, 4, .2),
    #         (8, 18, .1),
    #         (9, 18, .1),
    #         (9, 28, .1),
    #         (10, 28, .1)
    #     ],
    #     # Proximity
    #     [
    #         (0, .85, .1),
    #         (10, .85, .1)
    #     ]
    # ],
    # [
    #     # narrative
    #     [
    #         (0, 26, 'Owner is at Desk'),
    #         (2, 26, 'Owner Leaves Desk'),
    #         (4, 26, 'Owner Returns to Desk, Clients Arrive'),
    #         (6, 26, 'Owner Visits Neighbour'),
    #         (8, 26, 'Owner Returns to Desk'),
    #     ],
    #     # Alpha
    #     [
    #         (0, 100),
    #         (10, 100)
    #     ],
    #     # Privacy
    #     [
    #         (0, 20, .2),
    #         (10, 20, .2)
    #     ],
    #     # Unfamiliarity
    #     [
    #         (0, 0, .2),
    #         (4, 0, .2),
    #         (4, 4, .2),
    #         (7, 4, .2),
    #         (7, 0, .2),
    #         (10, 0, .2)
    #     ],
    #     # Proximity
    #     [
    #         (0, .75, .2),
    #         (2, .75, .2),
    #         (2, 0, .1),
    #         (4, 0, .1),
    #         (4, .75, .2),
    #         (6, .75, .2),
    #         (6, .5, .5),
    #         (8, .5, .5),
    #         (8, .75, .2),
    #         (10, .75, .2)
    #     ]
    # ],
    [
        [],
        # Alpha
        [
            (0, 3),
            (2.5, 3),
            (2.5, 25),
            (5, 25),
            (5, 100),
            (7.5, 100),
            (7.5, 200),
            (10, 200)
        ],
        # Privacy
        [
            (x / 20, max((math.sin((x - 9) / 8.25)) * 14 + (2 * random.random()) + 11, 0.1), 0) for x in range(201)
        ],
        # Unfamiliarity
        [
            (x / 20, max(math.sin((x - 9) / 8.25) * 7 + (1 * random.random()) + 5.5, 0.1), 0) for x in range(201)
        ],
        # Proximity
        [
            (x / 100, max(min((1.5 * math.sin(x / 12) + .75) / 2 + (.5 * random.random() - .25), 1), 0), 0) for x in range(1001)
        ]
    ]
]
