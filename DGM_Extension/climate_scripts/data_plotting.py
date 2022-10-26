from dgm.utils.USHCN_handling import extract_time_and_value_arrays_from_GRU_data, plot_reformatted_data, load_raw_data


PATH_TO_DATA = "/home/philippe/Documents/Code/DGM_Extension/data/small_chunked_sporadic.csv"
PATH_TO_PLOTS = "/home/philippe/GRU_Playground"


def main():
    original_data = load_raw_data(PATH_TO_DATA)
    times, values = extract_time_and_value_arrays_from_GRU_data(original_data)
    plot_reformatted_data(times, values, PATH_TO_PLOTS)


if __name__ == "__main__":
    main()
