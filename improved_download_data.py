import asyncio
import aiohttp
import aiofiles
import logging
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path
from tqdm import tqdm
from aiohttp import ClientSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_url(base_url: str, layers: list, bbox: list, image_size: list, format: str = "image/png", crs: str = "EPSG:25832") -> str:
    layers_str = ",".join(layers)
    bbox_str = ",".join(map(str, bbox))
    width, height = image_size
    url = (f"{base_url}?VERSION=1.3.0&service=WMS&request=GetMap&Format={format}&"
           f"GetFeatureInfo=text/plain&CRS={crs}&Layers={layers_str}&BBox={bbox_str}&"
           f"width={width}&height={height}")
    return url

async def fetch_image(session, url: str, max_retries=5) -> bytes:
    attempt = 0
    while attempt < max_retries:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                logging.error(f"Failed to fetch image: HTTP status {response.status} on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Exception while fetching image on attempt {attempt + 1}: {e}")
        finally:
            attempt += 1
            await asyncio.sleep(0.2)
    logging.error(f"Failed to fetch image after {max_retries} attempts.")
    return None

def process_label_image(img_bytes: bytes):
    try:
        image = Image.open(BytesIO(img_bytes)).convert("L")
        image_array = np.array(image)
        road_pixels = image_array != 255
        road_percentage = np.mean(road_pixels)
        if road_percentage >= 0.01:
            return Image.fromarray(road_pixels.astype(np.uint8) * 255), True
        else:
            return None, False
    except Exception as e:
        logging.error(f"Error processing label image: {e}")
        return None, False

async def save_image(img_bytes: bytes, path: Path):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(img_bytes)
        logging.info(f"Saved image to {path}")

async def download_and_check_label(session, url: str, path: Path, label_semaphore: asyncio.Semaphore, label_delay: float, max_retries=5):
    async with label_semaphore:
        await asyncio.sleep(label_delay)
        attempt = 0
        while attempt < max_retries:
            try:
                label_bytes = await fetch_image(session, url)
                if label_bytes:
                    label_image, has_road = process_label_image(label_bytes)
                    if has_road and label_image:
                        label_image.save(path)
                        logging.info(f"Label meets criteria and is saved to {path}")
                        return True
                    return False
            except Exception as e:
                logging.error(f"Error processing label image on attempt {attempt + 1}: {e}")
            attempt += 1
            await asyncio.sleep(0.2)
        logging.error(f"Failed to download and check label after {max_retries} attempts.")
        return False

async def process_bbox_sequentially(session, bbox, base_label_url, base_image_url, label_folder, image_folder, preferred_image_size, existing_image_filenames, label_semaphore, label_delay, max_retries=5):
    label_url = get_url(base_label_url, ["bygning"], bbox, preferred_image_size)
    image_url = get_url(base_image_url, ["ortofoto"], bbox, preferred_image_size)

    filename = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"

    if filename in existing_image_filenames:
        logging.info(f"Skipping {filename} as it already exists")
        return

    label_path = label_folder / filename
    image_path = image_folder / filename

    # Download and check the label, with rate limiting
    label_success = await download_and_check_label(session, label_url, label_path, label_semaphore, label_delay, max_retries)
    if label_success:
        # Fetch and save the image concurrently
        image_bytes = await fetch_image(session, image_url)
        if image_bytes:
            await save_image(image_bytes, image_path)

async def process_bbox_concurrently(session, bbox, base_label_url, base_image_url, label_folder, image_folder, preferred_image_size, existing_image_filenames, label_semaphore, label_delay, semaphore, max_retries=5):
    async with semaphore:
        await process_bbox_sequentially(session, bbox, base_label_url, base_image_url, label_folder, image_folder, preferred_image_size, existing_image_filenames, label_semaphore, label_delay, max_retries)

async def main():
    # Setup your parameters
    starting_point = [272038, 6656016]
    ending_point = [320189, 6700359]
    preferred_image_size = [1024, 1024]
    resolution = 0.1  # Each pixel represents 0.1 units
    label_delay = 0.01  # Delay between label requests in seconds
    max_concurrent_tasks = 20  # Adjust as needed

    bbox_size = [preferred_image_size[0] * resolution, preferred_image_size[1] * resolution]

    data_folder = Path("data").joinpath(f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}")
    image_folder = data_folder / "images"
    label_folder = data_folder / "labels"
    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)

    base_label_url = "https://openwms.statkart.no/skwms1/wms.fkb"
    base_image_url = "https://wms.geonorge.no/skwms1/wms.nib"

    # Calculate step sizes for 50% overlap
    step_size_x = bbox_size[0] / 2
    step_size_y = bbox_size[1] / 2

    # Generate x and y positions
    x_positions = []
    current_x = starting_point[0]
    while current_x <= ending_point[0] - bbox_size[0]:
        x_positions.append(current_x)
        current_x += step_size_x
    # Ensure we cover the ending_point
    if x_positions[-1] + bbox_size[0] < ending_point[0]:
        x_positions.append(ending_point[0] - bbox_size[0])

    y_positions = []
    current_y = starting_point[1]
    while current_y <= ending_point[1] - bbox_size[1]:
        y_positions.append(current_y)
        current_y += step_size_y
    # Ensure we cover the ending_point
    if y_positions[-1] + bbox_size[1] < ending_point[1]:
        y_positions.append(ending_point[1] - bbox_size[1])

    # Create bounding boxes with 50% overlap
    bboxes = [
        [x, y, x + bbox_size[0], y + bbox_size[1]]
        for x in x_positions for y in y_positions
    ]

    pbar = tqdm(total=len(bboxes), desc="Processing BBoxes")

    existing_image_filenames = set([x.name for x in image_folder.glob("*.png")])

    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Limits concurrent tasks
    label_semaphore = asyncio.Semaphore(1)  # Only one label request at a time

    async with ClientSession() as session:
        tasks = [
            process_bbox_concurrently(
                session, bbox, base_label_url, base_image_url, label_folder,
                image_folder, preferred_image_size, existing_image_filenames,
                label_semaphore, label_delay, semaphore
            )
            for bbox in bboxes
        ]

        for f in asyncio.as_completed(tasks):
            await f
            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    asyncio.run(main())
