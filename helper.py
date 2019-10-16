def print_importances(model, selected_cols):
    weigts_sum = sum(map(abs, model.coef_))
    for name, weight in sorted(zip(selected_cols, model.coef_), key=lambda x: -abs(x[1])):
        percent_weight = abs(weight) / weigts_sum
        print('{:40} {:.2%} {:15.2}'.format(name, percent_weight, weight))