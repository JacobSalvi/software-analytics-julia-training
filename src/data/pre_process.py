


def process_data(data):
    """
    Pre-process the data
    """
    # Drop missing values
    data = data.dropna()

    # Drop duplicates
    data = data.drop_duplicates()

    return data
