from ipywidgets import widgets, HBox
from IPython import display
import matplotlib.pyplot as plt
import random
from skimage import measure, exposure, color, segmentation
from rasterio import features
import shapely
from shapely.geometry import box, shape
import numpy as np
import requests
import json
import os
from gbdxtools import CatalogImage

def plot_array(array, subplot_ijk, title="", font_size=18, cmap=None):
    sp = plt.subplot(*subplot_ijk)
    sp.set_title(title, fontsize=font_size)
    plt.axis('off')
    plt.imshow(array, cmap=cmap)


def from_geojson(source):
    if source.startswith('http'):
        response = requests.get(source)
        geojson = json.loads(response.content)
    else:
        if os.path.exists(source):
            with open(source, 'r') as f:
                geojson = json.loads(f.read())
        else:
            raise ValueError("File does not exist: {}".format(source))

    geometries = []
    feats = []
    for f in geojson['features']:
        geom = shape(f['geometry'])
        feats.append({'geometry': geom, 'properties': {}})
        geometries.append(geom)

    return geometries, feats


def create_chip_boundaries(img, tile_size_pixels, overlap_pixels=0, mask_geom=None):
    tile_size_degrees = tile_size_pixels * img.affine.a
    overlap_degrees = overlap_pixels * img.affine.a

    xmin, ymin, xmax, ymax = img.bounds

    xcoords = np.arange(xmin, xmax + tile_size_degrees, tile_size_degrees - 2 * overlap_degrees)
    ycoords = np.arange(ymin, ymax + tile_size_degrees, tile_size_degrees - 2 * overlap_degrees)

    xmins = xcoords[0:-1]
    xmaxs = xcoords[1:]
    ymins = ycoords[0:-1]
    ymaxs = ycoords[1:]

    xpairs = np.column_stack([xmins, xmaxs])
    ypairs = np.column_stack([ymins, ymaxs])
    # add the overlap
    xpairs = xpairs + np.array([-overlap_degrees, overlap_degrees])[None, :] + overlap_degrees
    ypairs = ypairs + np.array([-overlap_degrees, overlap_degrees])[None, :] + overlap_degrees

    chip_bounds = []
    for xx in xpairs:
        for yy in ypairs:
            chip_bounds.append([xx[0], yy[0], xx[1], yy[1]])

    chip_geoms = [box(*tb) for tb in chip_bounds if box(*tb).intersects(box(*img.bounds))]

    if mask_geom is not None:
        chip_geoms = [g for g in chip_geoms if g.intersects(mask_geom)]

    return chip_geoms


class LabelSegment(object):
    def __init__(self, regionprops, image, segmented_image):

        self.regionprops = regionprops
        self.image = image
        self.segmented_image = segmented_image
        self.id = self.regionprops.label

        self.__validate_raw__()

        self.catid = image.cat_id
        self.label_value = None

    def __validate_raw__(self):
        if type(self.regionprops) != measure._regionprops._RegionProperties:
            raise TypeError("regionprops is not an instance of skimage.measure._regionprops._RegionProperties")

    def buffered_bbox(self, pixel_buffer=50):

        bbox = self.regionprops.bbox
        # sometimes this is misordered, so try to fix
        bbox_row_start = bbox[0]
        bbox_row_stop = bbox[2]
        bbox_col_start = bbox[1]
        bbox_col_stop = bbox[3]

        row_start = int(np.maximum(bbox_row_start - pixel_buffer, 0))
        row_stop = int(np.minimum(bbox_row_stop + pixel_buffer, self.image.shape[1]))
        col_start = int(np.maximum(bbox_col_start - pixel_buffer, 0))
        col_stop = int(np.minimum(bbox_col_stop + pixel_buffer, self.image.shape[2]))

        return (row_start, row_stop, col_start, col_stop)

    def rgb(self, pixel_buffer=50, blm=True):

        row_start, row_stop, col_start, col_stop = self.buffered_bbox(pixel_buffer)

        return self.image.base_layer_match(blm=blm)[row_start:row_stop, col_start:col_stop]

    def pan(self, pixel_buffer=50, equalize_histogram=True):

        pan = color.rgb2gray(self.rgb(pixel_buffer))
        if equalize_histogram is True:
            pan = exposure.equalize_hist(pan)

        return pan

    def binary(self, pixel_buffer=50):

        row_start, row_stop, col_start, col_stop = self.buffered_bbox(pixel_buffer)
        binary = (self.segmented_image == self.regionprops.label)[row_start:row_stop, col_start:col_stop]

        return binary

    def mark_boundaries(self, pixel_buffer=50):

        return segmentation.mark_boundaries(self.rgb(pixel_buffer), self.binary(pixel_buffer))

    def set_label_value(self, value):

        assert (type(value) is bool or value is None), 'label value must boolean'

        self.label_value = value

    def _image(self, pixel_buffer=0):

        row_start, row_stop, col_start, col_stop = self.buffered_bbox(pixel_buffer)

        return np.rollaxis(self.image[:, row_start:row_stop, col_start:col_stop], 0, 3)

    def _window(self, pixel_buffer=0):

        row_start, row_stop, col_start, col_stop = self.buffered_bbox(pixel_buffer)

        return self.image[:, row_start:row_stop, col_start:col_stop]

    def masked_image(self, pixel_buffer=0):

        img = self._image(pixel_buffer)
        return np.ma.masked_where(np.broadcast_to(self.binary(pixel_buffer)[:, :, None], img.shape) == 0, img)

    def __to_polygon__(self):

        # create polygon generator object
        segment = self.segmented_image == self.id
        polygon_generator = features.shapes(segment.astype('uint8'),
                                            mask=segment != 0,
                                            transform=self.image.affine)
        # Extract out the individual polygons, fixing any invald geometries using buffer(0)
        polygon = [shape(g).buffer(0) for g, v in polygon_generator][0]

        return polygon

    def to_veda(self, pixel_buffer):
        geom = self.__to_polygon__()
        return (self._window(pixel_buffer), geom)

    def __as_geojson__(self):
        d = {'geometry': self.__to_polygon__().__geo_interface__,
             'properties': {'catalog_id': self.catid,
                            'bbox': self.image.bounds,
                            'label_value': self.label_value,
                            'id': self.id,
                            'img_options': dict(self.image.options)}
             }

        return d



class LabelPolygon(LabelSegment):
    def __init__(self, geom, id, image):

        self.geom = geom
        self.image = image
        self.id = id

        self.__validate_raw__()

        self.catid = image.cat_id
        self.label_value = None

        # convert to array and then region props
        self.segmented_image = self.__to_segments__()
        if np.sum(self.segmented_image) == 0:
            # the geometry doesn't intersect the footprints, so raise value error
            raise ValueError("Geometry does not intersect image boundaries.")
        self.regionprops = measure.regionprops(self.segmented_image)[0]

    def __validate_raw__(self):
        if type(self.geom) != shapely.geometry.polygon.Polygon:
            raise TypeError("geom is not an instance of shapely.geometry.polygon.Polygon")

    def __to_segments__(self):

        geom_array = features.rasterize([(self.geom, 1)],
                                        out_shape=(self.image.shape[1], self.image.shape[2]),
                                        transform=self.image.affine,
                                        fill=0,
                                        all_touched=True,
                                        dtype=np.uint8)

        return geom_array

    def to_veda(self, pixel_buffer):
        return (self._window(pixel_buffer), self.geom)

    def __to_polygon__(self):
        return self.geom


class LabelData(object):
    def __init__(self, features=None, image=None, description=None):

        self.features = features
        self.image = image
        self.description = description

        self.__validate_features__()
        self.__validate_image__()
        self.__validate_description__()

        self.__populate__()
        self.index = 0

    def __getitem__(self, key):
        value = getattr(self, key, None)
        if value is not None:
            return value
        else:  # backwards compatability
            return getattr(self, PROPS[key])

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        try:
            result = self.data[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.data)

    def __validate_features__(self):

        if self.features is None:
            return

        # validate that features is of the correct type
        is_polygon_list = False
        is_label_array = False
        if type(self.features) == list:
            if np.all([type(f) == shapely.geometry.polygon.Polygon for f in self.features]):
                is_polygon_list = True
        elif type(self.features) == np.ndarray:
            if self.features.dtype == np.dtype('int'):
                is_label_array = True

        if is_polygon_list is False and is_label_array is False:
            err = '''Invalid input for features. Must be either a list of shapely polygon geometries or 
                     a numpy integer array.'''
            raise TypeError(err)

        if is_polygon_list:
            self.feature_type = 'polygons'
            self.n_features = len(self.features)
        elif is_label_array:
            self.feature_type = 'label_array'
            unique_vals = list(np.unique(self.features))
            # drop zeros (these get ignored)
            if 0 in unique_vals:
                unique_vals.remove(unique_vals.index(0))
            self.n_features = len(unique_vals)

    def __validate_image__(self):

        if self.features is None:
            return

        if hasattr(self.image, 'ipe') is False:
            raise TypeError("One or more input images are not gbdxtools Image objects")

    def __validate_description__(self):

        if self.features is None:
            return

        if type(self.description) != str:
            raise TypeError("Input label must be of type string")

    def __chips__(self, chip_shape=(256, 256), chip_offset_rows=0, chip_offset_cols=0):

        if chip_offset_rows < 0:
            raise ValueError("chip_offset_rows must be >= 0")
        if chip_offset_cols < 0:
            raise ValueError("chip_offset_cols must be >= 0")

        nrows = self.image.shape[1]
        chip_height = chip_shape[0]

        ncols = self.image.shape[2]
        chip_width = chip_shape[1]

        chip_col_starts = range(chip_offset_cols, ncols + 1, chip_width)
        chip_row_starts = range(chip_offset_rows, nrows + 1, chip_height)
        chips = []
        for chip_col_start in chip_col_starts:
            chip_col_stop = chip_col_start + chip_width
            for chip_row_start in chip_row_starts:
                chip_row_stop = chip_row_start + chip_height
                chip = (chip_row_start, chip_row_stop, chip_col_start, chip_col_stop)
                chips.append(chip)

        return chips

    def features_to_veda(self, pixel_buffer=0, skip_nulls=True):

        if skip_nulls is True:
            return [d.to_veda(pixel_buffer) for d in self.data if d.label_value is not None]
        else:
            return [d.to_veda(pixel_buffer) for d in self.data]

    def chips_to_veda(self, skip_nulls=True, chip_shape=(256, 256), chip_offset_rows=0, chip_offset_cols=0,
                      out_format='geometry'):

        if out_format not in ['array', 'geometry']:
            raise ValueError("out_format must be one of [array, geometry]")

        # calculate the chip indices
        chips = self.__chips__(chip_shape=chip_shape, chip_offset_rows=chip_offset_rows,
                               chip_offset_cols=chip_offset_cols)
        outputs = []
        for chip in chips:
            # subset the image to the chip boundaries
            i, ii, j, jj = chip
            chip_image = self.image[:, i:ii, j:jj]
            if self.feature_type == 'polygons':
                # find all features that intersect the chip
                chip_features = [f for f in self.data if f.geom.intersects(chip_image.asShape())]
                # extract out the nulls
                null_features = [1 for f in chip_features if f.label_value is None]
                # check that either there are no null (unlabeled) features in this chip
                # or skip_nulls is False
                if skip_nulls is False or len(null_features) == 0:
                    # extract out the positives
                    positive_features = [(f.geom, 1) for f in chip_features if f.label_value is True]
                    if out_format == 'geometry':
                        outputs.append((chip_image, positive_features))
                    elif out_format == 'array':
                        if len(positive_features) > 0:
                            # convert from geom to array using the chip affine
                            geom_array = features.rasterize(positive_features,
                                                            out_shape=(chip_image.shape[1], chip_image.shape[2]),
                                                            transform=chip_image.affine,
                                                            fill=0,
                                                            all_touched=True,
                                                            dtype=np.uint8)
                        else:
                            # if no positive features, the whole chip is zero
                            geom_array = np.zeros((chip_image.shape[1], chip_image.shape[2]), dtype=np.uint8)
                        # cehck that the geom_array is the correct size (it can be clipped at image edge)
                        if chip_image.shape[1:] == chip_shape:
                            # package up the binary array and chip image
                            outputs.append((chip_image, geom_array))
            elif self.feature_type == 'label_array':
                # subsetting self.features to just the current chip area
                chip_features = self.features[i:ii, j:jj]
                # extract out the null (unlabeled) areas
                null_labels = [d.regionprops.label for d in self.data if d.label_value is None]
                # check for nulls in the chip
                nulls_in_chip = np.any(np.isin(chip_features.ravel(), null_labels, assume_unique=False,
                                               invert=False))
                if skip_nulls is False or nulls_in_chip is False:
                    if out_format == 'geometry':
                        positive_features = [d.__to_polygon__() for d in self.data if d.label_value is True]
                    elif out_format == 'array':
                        # find the ones that are positives
                        positive_labels = [d.regionprops.label for d in self.data if d.label_value is True]
                        # convert the features to a binary array, including only the positive labels
                        positive_features = np.isin(chip_features.ravel(), positive_labels, assume_unique=False,
                                                    invert=False).reshape(chip_features.shape).astype(np.uint8)
                    # verify that the chip is the correct shape before adding to results
                    # (it can be clipped at image edge)
                    if chip_image.shape[1:] == chip_shape:
                        # package up the binary array and the chip image
                        outputs.append((chip_image, positive_features))

        return outputs

    def __populate__(self):

        if self.features is None or self.image is None:
            return

        if self.feature_type == 'label_array':
            # verify that the label array and input image are the same size
            if self.features.shape != self.image[0, :, :].shape:
                raise ValueError("Shape mismatch between features and image.")
            self.data = [LabelSegment(regionprops, self.image, self.features) for regionprops in
                         measure.regionprops(self.features)]
        elif self.feature_type == 'polygons':
            self.data = [LabelPolygon(geom, i, self.image) for i, geom in enumerate(self.features) if
                         geom.intersects(self.image.asShape())]
            self.n_features = len(self.data)

    def to_geojson(self, filename):

        out_features = [feat.__as_geojson__() for feat in self.data]
        with open(filename, 'w') as f:
            f.write(json.dumps(out_features))

    def from_geojson(self, filename):
        with open(filename, 'r') as f:
            in_features = json.loads(f.read())

        catids = list(set([feat['properties']['catalog_id'] for feat in in_features]))
        if len(catids) > 1:
            raise ValueError("Input geojson references multiple catalog_ids")
        else:
            catid = catids[0]

        bboxes = list(set([feat['properties']['bbox'] for feat in in_features]))
        if len(bboxes) > 1:
            raise ValueError("Input geojson references multiple bboxes")
        else:
            bbox = bboxes[0]

        options_list = list(set([feat['properties']['img_options'] for feat in in_features]))
        if len(options_list) > 1:
            raise ValueError("Input geojson references multiple img_options")
        else:
            options = options_list[0]

        self.image = CatalogImage(catid, bbox=bbox, **options)
        self.__validate_image__()

        for feat in in_features:
            geom = shape(feat['geometry'])
            self.features.append(geom)
            label_polygon = LabelPolygon(geom, feat['id'], self.image)
            label_polygon.label_value = feat['label_value']
            self.data.append(label_polygon)
        self.n_features = len(self.data)
        self.__validate_features__()
        self.index = 0


class LabelWidget(object):
    def __init__(self, text="Is this a feature of interest?", figsize=(20, 20), shuffled=False, show_src_img=False,
                 pixel_buffer=50):

        self.text = text
        self.figsize = figsize
        self.label_data = []
        self.tally = 0
        self.results = []
        self.shuffled = shuffled
        self.show_src_img = show_src_img
        self.pixel_buffer = pixel_buffer

        self.__create_vote_buttons__()
        self.buttons = self.__add_button_callback__(['Yes', 'No', 'Skip'], self.__catch_vote_and_advance__)
        self.buttons = self.__add_button_callback__(['Back'], self.__back__)
        self.buttons = self.__add_button_callback__(['Clear All'], self.__reset__)

    def add_data(self, label_data):

        if type(label_data) == LabelData:
            if self.shuffled is True:
                # shuffle the input label data, and append to the end of the current list
                _label_data = random.sample(label_data, len(label_data))
            else:
                _label_data = label_data
            self.label_data.extend(_label_data)
            # find out how many have been previously labeled
            previously_labeled = [d for d in self.label_data if d.label_value is not None]
            self.tally = len(previously_labeled)
            # split the data between the labeled and unlabeled
            unlabeled = [d for d in self.label_data if d.label_value is None]
            # reorder the data so that the previously labeled are at the beginning
            reordered = previously_labeled + unlabeled
            self.label_data = reordered
        else:
            raise TypeError("label_data is not an instance of LabelData")

    def __create_vote_buttons__(self):

        button_yes = widgets.Button(description='Yes')
        button_no = widgets.Button(description='No')
        button_skip = widgets.Button(description='Skip')
        button_back = widgets.Button(description='Back')
        button_reset = widgets.Button(description='Clear All')
        self.buttons = HBox([button_yes, button_no, button_skip, button_back, button_reset])

    def initialize_voting(self):

        display.clear_output(wait=True)
        feature = self.label_data[self.tally]
        self.__display_feature__(feature)
        display.display(self.buttons)

    @staticmethod
    def __plot_array__(array, subplot_ijk, title="", font_size=18, cmap=None):
        sp = plt.subplot(*subplot_ijk)
        sp.set_title(title, fontsize=font_size)
        plt.axis('off')
        plt.imshow(array, cmap=cmap)

    def __display_feature__(self, feature):
        sp = plt.figure(figsize=self.figsize)
        if self.show_src_img is True:
            self.__plot_array__(feature.mark_boundaries(pixel_buffer=self.pixel_buffer), (1, 2, 1), title=self.text)
            self.__plot_array__(feature.rgb(pixel_buffer=self.pixel_buffer), (1, 2, 2), title='Source Image')
        else:
            self.__plot_array__(feature.mark_boundaries(pixel_buffer=self.pixel_buffer), (1, 1, 1), title=self.text)

        _ = plt.plot()
        status_msg = "Labeled {tally} out of {total} features.".format(tally=self.tally,
                                                                       total=len(self.label_data))
        print(status_msg)

    @staticmethod
    def convert_response_to_binary(response):
        if response == 'Yes':
            binary = True
        elif response == 'No':
            binary = False
        elif response == 'Skip':
            binary = None

        return binary

    def __catch_vote_and_advance__(self, b):

        # record the vote to the existing feature
        vote = self.convert_response_to_binary(b.description)
        current_segment = self.label_data[self.tally]
        current_segment.set_label_value(vote)

        # increment the tally
        self.tally += 1

        # exit if no features remain
        if self.tally == len(self.label_data):
            b.close_all()
            display.clear_output(wait=True)
            print("All data has been labeled!")
            return


        # otherwise, advance to the next feature
        display.clear_output(wait=True)
        next_segment = self.label_data[self.tally]
        self.__display_feature__(next_segment)
        display.display(self.buttons)

    def __back__(self, b):

        # decrement the tally
        self.tally -= 1

        # change the feature label back to none
        previous_segment = self.label_data[self.tally]
        previous_segment.set_label_value(None)

        # otherwise, advance to the next feature
        display.clear_output(wait=True)
        self.__display_feature__(previous_segment)
        display.display(self.buttons)

    def __reset__(self, b):

        # decrement the tally
        self.tally = 0

        # reset all of hte values
        for d in self.label_data:
            d.set_label_value(None)

        self.initialize_voting()

    def __add_button_callback__(self, descriptions, callback):
        for b in self.buttons.children:
            if b.description in descriptions:
                b.on_click(callback)

        return self.buttons
