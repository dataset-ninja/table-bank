import os
import shutil
from collections import defaultdict

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    images_path = "/home/alex/DATASETS/TODO/TableBank/TableBank/Detection/images"
    batch_size = 30
    ann_jsons_path = "/home/alex/DATASETS/TODO/TableBank/TableBank/Detection/annotations"

    ds_name_to_jsons = {
        "train": ("tablebank_word_train.json", "tablebank_latex_train.json"),
        "val": ("tablebank_latex_val.json", "tablebank_word_val.json"),
        "test": ("tablebank_latex_test.json", "tablebank_word_test.json"),
    }

    def create_ann(image_path):
        labels = []

        image_name = get_file_name_with_ext(image_path)
        img_height = image_name_to_shape[image_name][0]
        img_wight = image_name_to_shape[image_name][1]

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            polygons_coords = curr_ann_data[0]
            for coords in polygons_coords:
                exterior = []
                for i in range(0, len(coords), 2):
                    exterior.append([int(coords[i + 1]), int(coords[i])])
                if len(exterior) < 3:
                    continue
                poligon = sly.Polygon(exterior)
                label_poly = sly.Label(poligon, obj_class)
                labels.append(label_poly)

            bbox_coord = curr_ann_data[1]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, obj_class)
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    obj_class = sly.ObjClass("table", sly.AnyGeometry, color=(255, 0, 0))
    latex_meta = sly.TagMeta("latex", sly.TagValueType.NONE)
    word_meta = sly.TagMeta("word", sly.TagValueType.NONE)

    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[latex_meta, word_meta])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, json_names in ds_name_to_jsons.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for idx, curr_json_name in enumerate(json_names):
            if idx == 0:
                tag = sly.Tag(latex_meta)
            else:
                tag = sly.Tag(word_meta)

            ann_json_path = os.path.join(ann_jsons_path, curr_json_name)

            ann = load_json_file(ann_json_path)

            image_id_to_name = {}
            image_name_to_ann_data = defaultdict(list)
            image_name_to_shape = {}

            for curr_image_info in ann["images"]:
                image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]
                image_name_to_shape[curr_image_info["file_name"]] = (
                    curr_image_info["height"],
                    curr_image_info["width"],
                )

            for curr_ann_data in ann["annotations"]:
                image_id = curr_ann_data["image_id"]
                image_name_to_ann_data[image_id_to_name[image_id]].append(
                    [curr_ann_data["segmentation"], curr_ann_data["bbox"]]
                )

            images_names_json = list(image_name_to_ann_data.keys())
            real_names = os.listdir(images_path)
            images_names = list(set(images_names_json) & set(real_names))
            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                images_pathes_batch = [
                    os.path.join(images_path, image_path) for image_path in img_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns_batch)

                progress.iters_done_report(len(img_names_batch))

    return project
