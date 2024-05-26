import cv2
from numpy import array

from web_interaction import image_parser as ip


def test_get_backlog_description():
    frame = cv2.imread("tests/test_images/iframe_backlog.png")
    backlog = ip.get_backlog(frame)
    assert len(backlog) == 6
    assert len(backlog[0]) == 3
    expected_backlog = [
        (array([43, 194, 249]), 8.0, (725, 183)),
        (array([43, 194, 249]), 8.0, (771, 183)),
        (array([43, 194, 249]), 13.0, (725, 229)),
        (array([43, 194, 249]), 9.0, (771, 229)),
        (array([120, 79, 240]), 17.0, (725, 275)),
        (array([120, 79, 240]), 5.0, (771, 275)),
    ]

    for expected, actual in zip(expected_backlog, backlog):
        expected_color, expected_hours, expected_position = expected
        color, hours, position = actual
        assert color == frozenset(enumerate(expected_color))
        assert hours == expected_hours
        assert position == expected_position


def test_get_user_stories():
    image = cv2.imread("tests/test_images/iframe_user_stories.png")
    user_stories, positions = ip.get_user_stories(image)
    assert len(user_stories) == 5
    assert len(user_stories[0]) == 3
    expected_user_stories = [
        (array([23, 150, 247]), 0.05, 2.0),
        (array([43, 194, 249]), 0.065, 1.0),
        (array([255, 211, 143]), 0.045, 1.0),
        (array([120, 79, 240]), 0.08, 2.0),
        (array([115, 188, 30]), 0.07, 2.0),
    ]
    expected_positions = [(725, 183), (725, 229), (725, 275), (725, 321), (725, 367)]
    assert positions == expected_positions
    for expected, actual in zip(expected_user_stories, user_stories):
        color, loyalty, customers = actual
        ex_color, ex_loyalty, ex_customers = expected
        assert color == frozenset(enumerate(ex_color))
        assert loyalty == ex_loyalty
        assert customers == ex_customers
