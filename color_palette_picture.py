import numpy as np
import pandas as pd
from PIL import Image
import sys

def count_colors(array_picture: np.array) -> pd.DataFrame:
    '''
    this method returns a pd.DataFrame which holds the count of all the single rgb colors detected in the picture given as input
    :param array_picture: numpy.ndarray of the picture
    :return: pd.DataFrame of the rgb colors detected, with the number of occurences of each color
    '''

    # squeezing the 3D array down to a 1D array which holds only the tuples of rgb colors
    value, count = np.unique(array_picture.reshape(-1, 3), axis=0, return_counts=True)

    rgb_count_dict = {tuple(zipped[0]): zipped[1] for zipped in zip(value, count)}

    df_colors = pd.DataFrame(data={
        'colors': rgb_count_dict.keys(),
        'count': rgb_count_dict.values()
    }).sort_values('count', ascending=False, ignore_index=True)

    return df_colors

def debug_dec(func):
    '''
    decorator to be used to debug a function. It prints out the args passed and the outputs
    :param func: function to be decorated and therefor debugged
    :return: None
    '''
    def wrapper(*args, **kwargs):
        print(f'****** {func.__name__} ******')
        print(f'Args: {args}')
        print(f'Kwargs: {kwargs}')
        print(f'Result: {func(*args, **kwargs)}')
        print(f'****** END - {func.__name__} ******')
        return func(*args, **kwargs)
    return wrapper


def distance_vect(coord1: tuple, coord2: tuple) -> float:
    '''
    method to calculate the distance between two coordinates in a 3d space
    :param coord1: coordinate of the first point
    :param coord2: coordinate of the second point
    :return: distance in a 3d space
    '''
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)


def is_in_color_cluster(df_colors: tuple, rgb_color: tuple, distance: int) -> pd.DataFrame:
    '''
    the method calcuates whether an rgb color is in the cluster of the given rgb_color passed as param, basing the rating on the param passed as distance
    :param df_colors: tuple of rgb color taken from a pandas dataframe
    :param rgb_color: tuple describing the rgb color which will be center of the cluster
    :return: boolean mask for the given color cluster to apply on the original pd.DataFrame counting the occurrences of each color
    '''
    return True if distance_vect(rgb_color, df_colors) < distance else False


def color_palette(df_colors: pd.DataFrame, nr_colors: int = 5, cluster_radius: int = 50) -> dict:
    '''
    the method return a dict of the first 'nr_colors' detected in the picture.
    The method searches for the top rgb colors in terms of occurences in the picture, then filters the df_colors dataframe in order to strip out the values which are
    rated similar based on the value 'color likeliness' (the higher the value, the more the cluster around the color chosen will be wide)
    :param df_colors: pandas dataframe with the rgb colors contained in a picture and their occurence
    :param nr_colors: number of colors which are required to build the color palette, default 5 (as adobe color site)
    :param cluster_radius: radius of the cluster in points of color (default 50). The higher the radius the wider the cluster
    :return: dictionary of the first 'nr_colors' detected in the picture with the list of rgb colors in the cluster
    '''

    output_dict = {}
    try:
        while len(output_dict) < nr_colors:
            # defining the color with the most occurences in the array
            index_color = df_colors['count'].idxmax()
            color = df_colors['colors'].iloc[index_color]
            cluster_boolean_mask = df_colors['colors'].apply(is_in_color_cluster,
                                                             args=(color, cluster_radius))  # boolean pd.Series
            cluster_colors: list = list(df_colors['colors'].loc[cluster_boolean_mask])
            output_dict[color] = cluster_colors

            # drop the colors in df_colors, which have been grouped in the previous cluster
            df_colors = df_colors[~cluster_boolean_mask]  # the tilda mark, inverts the boolean values of the series.
        return output_dict
    except IndexError as ind_err:
        return output_dict

def palettize_pic(picture, nr_colors_palette: int = 10, cluster_radius: int = 50) -> dict:
    '''
    this method takes a picture as input and a dictionary of the color extracted from the picture and returns a palettized versiion of the picture saved in the working directory
    :param pic_array: array of rgb colors representing the picture or path to the picture, or path to the picture to process
    :param nr_colors_palette: number of colors in the palette to extract (default 10)
    :param cluster_radius: radius of the cluster to be considered for the cluster (default 50)
    :return: picture processed and palettized as np.ndarray AND dictionary of the colors extracted from the picture as key of a dictionary containing the rgb colors consider in that cluster as values
    '''
    if type(picture) == str:
        try:
            pic_array = np.asarray(Image.open(picture)).astype(dtype=np.int32)
            # mi accerto di avere array di dimensioni (width, height, 3)
            if pic_array.shape[2] != 3:
                diff = pic_array.shape[2] - 3
                pic_array = np.delete(pic_array, range(3, 3 + diff), axis=2)
        except FileNotFoundError as err:
            print('File not found...\n\t\t', err)
            sys.exit(1)
    if type(picture) == np.ndarray:
        pic_array = picture

    # first of all the function counts the number of unique rgb colors in the picture
    df_colors_unique = count_colors(pic_array)
    # then the unique colors are used to extract a palette in form of a dictionary containing the rgb color representing the cluster as key, and all of the colors in the cluster as values
    palette_dict: dict = color_palette(df_colors_unique,
                                       nr_colors=40,
                                       cluster_radius=50)

    for color in palette_dict.keys():
        squared_dist = (pic_array - np.array(color)) ** 2
        mask = np.sqrt(squared_dist.sum(axis=-1)) < cluster_radius
        pic_array[np.where(mask == True)] = color
    pict_to_save = pic_array.astype(dtype=np.uint8)  # PIL accepts only uint8 if detects an array of shape (x,y,3)
    path_to_save = input('Enter the name of the file to save the processed picture:')
    Image.fromarray(pict_to_save).save(f'./{path_to_save}.jpeg')
    return pict_to_save, palette_dict


def focus_points(starting_array: np.array, perc_focus: float = 0.06) -> list:
    '''
    method to get 4 array of pixels around the 4 focal points of the rule of thirds
    :param starting_array: initial array
    :param perc_focus: percentage of the area covered by one of the 4 new focal areas
    :return: list of array containing the pixels of the 4 focal areas
    '''
    width = starting_array.shape[0]
    height = starting_array.shape[1]
    area = width * height
    area_of_sample = area * perc_focus
    sample_side_length = int(np.sqrt(area_of_sample))

    width_thirds = (width // 3, width // 3 * 2)
    height_thirds = (height // 3, height // 3 * 2)
    thirds_array = np.zeros(shape=(4, 2))

    for index, values in enumerate(product(width_thirds, height_thirds)):
        thirds_array[index] = values

    # define 4 areas around the focal points, based on the perc_focus value
    list_areas = []
    for row in thirds_array:
        # slices are defined to define the new areas around the focal points
        horizontal_slice = slice(int(row[0] - sample_side_length // 2), int(row[0] + sample_side_length // 2))
        vertical_slice = slice(int(row[1] - sample_side_length // 2), int(row[1] + sample_side_length // 2))
        list_areas.append(starting_array[horizontal_slice, vertical_slice])

    return list_areas

if __name__ == '__main__':
    picture_path = input('Enter the path of the picture to process:')
    processed_pic, palette = palettize_pic(picture=picture_path,
                                           nr_colors_palette=20,
                                           cluster_radius=50)
