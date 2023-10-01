import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox, Slider

from probabilistic_classification import get_probability_density_function, \
    generate_vectors, GaussParameters, get_interval, get_intersection, get_errors

COUNT = 10000
STEP = 0.001

SECOND_AREA_LABEL = 'Зона пропуска обнаружения для $C_1$'
FIRST_AREA_LABEL = 'Зона ложной тревоги для $C_1$'
SECOND_PLOT_LABEL = '$p(X | C_2) \cdot P(C_2)$'
FIRST_PLOT_LABEL = '$p(X | C_1) \cdot P(C_1)$'
LINE_LABEL = 'Оптимальное значение $X_m$'


def update_field(field, value, event_id=None, event=None):
    if event_id is not None:
        field.disconnect(event_id)
    field.set_val("{0:.2f}".format(value))
    if event_id is not None:
        return field.on_submit(event)


def get_annotation(detection_error, false_positive):
    detection_error_str = '{0:.2f}%'.format(detection_error * 100)
    false_positive_str = '{0:.2f}%'.format(false_positive * 100)
    summary_error_str = '{0:.2f}%'.format((false_positive + detection_error) * 100)
    annotation = '$P_{л.т.}=$' + detection_error_str + '\n$P_{п.о.}=$' + false_positive_str + \
                 '\n$P_{о.}=$' + summary_error_str
    return annotation


def main():
    matplotlib.use('TkAgg')

    plot = plt.subplot(111)
    plt.subplots_adjust(bottom=0.35)

    first_probability = 0.5
    second_probability = 0.5

    ax_first_mean = plt.axes((0.125, 0.15, 0.3, 0.075))
    ax_second_mean = plt.axes((0.6, 0.15, 0.3, 0.075))
    ax_first_probability = plt.axes((0.125, 0.25, 0.3, 0.075))
    ax_second_probability = plt.axes((0.6, 0.25, 0.3, 0.075))
    ax_first_std = plt.axes((0.125, 0.05, 0.3, 0.075))
    ax_second_std = plt.axes((0.6, 0.05, 0.3, 0.075))

    first_probability_field = TextBox(ax_first_probability, 'First probability')
    second_probability_field = TextBox(ax_second_probability, 'Second probability')
    first_mean_slider = Slider(ax_first_mean, 'First mean', -10, 0, valinit=-1)
    second_mean_slider = Slider(ax_second_mean, 'Second mean', 0, 10, valinit=1)
    first_std_field = TextBox(ax_first_std, 'First std')
    second_std_field = TextBox(ax_second_std, 'Second std')

    first_gauss_parameters = GaussParameters(-1, 1)
    second_gauss_parameters = GaussParameters(1, 1)

    def update_first_std(val):
        nonlocal first_gauss_parameters
        first_gauss_parameters.deviation = float(val)
        update(None)

    def update_second_std(val):
        nonlocal second_gauss_parameters
        second_gauss_parameters.deviation = float(val)
        update(None)

    def update_first_mean(val):
        nonlocal first_gauss_parameters
        first_gauss_parameters.mean = val
        update(None)

    def update_second_mean(val):
        nonlocal second_gauss_parameters
        second_gauss_parameters.mean = val
        update(None)

    def update_first_probability_field(text):
        nonlocal first_probability, second_probability, second_probability_submit_id
        first_probability = float(text)
        second_probability = 1 - first_probability
        second_probability_submit_id = update_field(second_probability_field, second_probability,
                                                    second_probability_submit_id, update_second_probability_field)
        update(None)

    def update_second_probability_field(text):
        nonlocal second_probability, first_probability, first_probability_submit_id
        second_probability = float(text)
        first_probability = 1 - second_probability
        first_probability_submit_id = update_field(first_probability_field, first_probability,
                                                   first_probability_submit_id, update_first_probability_field)
        update(None)

    def update(val):
        first_vector, second_vector = generate_vectors(first_gauss_parameters, second_gauss_parameters, COUNT)
        start, stop = get_interval(first_vector, second_vector)
        arrange = np.arange(start, stop, STEP)
        first_function_values = get_probability_density_function(first_vector, first_probability, arrange)
        second_function_values = get_probability_density_function(second_vector, second_probability, arrange)
        intersection = get_intersection(first_function_values, second_function_values)
        detection_error, false_positive = get_errors(first_function_values, second_function_values, intersection, STEP)
        annotation = get_annotation(detection_error, false_positive)
        plot.clear()
        plot.axvline(x=arrange[intersection], color='r', linestyle='--', label=LINE_LABEL)
        plot.plot(arrange, first_function_values, label=FIRST_PLOT_LABEL)
        plot.plot(arrange, second_function_values, label=SECOND_PLOT_LABEL)
        plot.fill_between(arrange[:intersection], second_function_values[:intersection], 0, alpha=0.5,
                          label=FIRST_AREA_LABEL)
        plot.fill_between(arrange[intersection:], first_function_values[intersection:], 0, alpha=0.5,
                          label=SECOND_AREA_LABEL)
        plot.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                      bbox=dict(boxstyle='round', fc='w'))
        plot.legend()
        plt.draw()

    first_probability_submit_id = first_probability_field.on_submit(update_first_probability_field)
    second_probability_submit_id = second_probability_field.on_submit(update_second_probability_field)
    first_mean_slider.on_changed(update_first_mean)
    second_mean_slider.on_changed(update_second_mean)
    first_std_field.on_submit(update_first_std)
    second_std_field.on_submit(update_second_std)

    first_probability_submit_id = update_field(first_probability_field, first_probability,
                                               first_probability_submit_id, update_first_probability_field)
    second_probability_submit_id = update_field(second_probability_field, second_probability,
                                                second_probability_submit_id, update_second_probability_field)
    update_field(first_std_field, first_gauss_parameters.deviation)
    update_field(second_std_field, second_gauss_parameters.deviation)
    update(None)
    plt.show()


if __name__ == '__main__':
    main()
