import asyncio
import re
from typing import Dict, List, Literal, Union

from openai import AsyncOpenAI, AsyncAzureOpenAI
from bs4 import BeautifulSoup

from image_generation.replicate import call_replicate


async def process_tasks(
    prompts: List[str],
    api_key: str | None,
    base_url: str | None,
    azure_openai_api_key: str | None = None,
    azure_openai_dalle3_api_version: str | None = None,
    azure_openai_resource_name: str | None = None,
    azure_openai_dalle3_deployment_name: str | None = None,
    model: Literal["dalle3", "flux"] = "dalle3",
):
    """
    Xử lý danh sách các prompt tạo ảnh.
    
    Nếu có cung cấp Azure OpenAI key, ưu tiên sử dụng Azure để tạo ảnh.
    Nếu không, sử dụng API key thông thường (cho trường hợp model "dalle3") hoặc chuyển sang replicate.
    """
    if azure_openai_api_key is not None:
        tasks = [
            generate_image_azure(
                prompt,
                azure_openai_api_key,
                azure_openai_dalle3_api_version,
                azure_openai_resource_name,
                azure_openai_dalle3_deployment_name,
            )
            for prompt in prompts
        ]
    elif api_key is not None and model == "dalle3":
        tasks = [generate_image_dalle(prompt, api_key, base_url) for prompt in prompts]
    else:
        tasks = [generate_image_replicate(prompt, api_key) for prompt in prompts]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results: List[Union[str, None]] = []
    for result in results:
        if isinstance(result, Exception):
            print(f"An exception occurred: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


async def generate_image_dalle(
    prompt: str, api_key: str, base_url: str | None
) -> Union[str, None]:
    """
    Tạo ảnh sử dụng DALL-E 3 thông qua OpenAI API.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    res = await client.images.generate(
        model="dall-e-3",
        quality="standard",
        style="natural",
        n=1,
        size="1024x1024",
        prompt=prompt,
    )
    await client.close()
    return res.data[0].url


async def generate_image_azure(
    prompt: str,
    azure_openai_api_key: str,
    azure_openai_api_version: str,
    azure_openai_resource_name: str,
    azure_openai_dalle3_deployment_name: str,
) -> str:
    """
    Tạo ảnh sử dụng DALL-E 3 thông qua Azure OpenAI.
    """
    client = AsyncAzureOpenAI(
        api_version=azure_openai_api_version,
        api_key=azure_openai_api_key,
        azure_endpoint=f"https://{azure_openai_resource_name}.openai.azure.com/",
        azure_deployment=azure_openai_dalle3_deployment_name,
    )
    image_params = {
        "model": "dall-e-3",
        "quality": "standard",
        "style": "natural",
        "n": 1,
        "size": "1024x1024",
        "prompt": prompt,
    }
    res = await client.images.generate(**image_params)
    await client.close()
    return res.data[0].url


async def generate_image_replicate(prompt: str, api_key: str) -> str:
    """
    Tạo ảnh thông qua dịch vụ Replicate (Flux Schnell).
    """
    return await call_replicate(
        {
            "prompt": prompt,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "output_quality": 100,
        },
        api_key,
    )


def extract_dimensions(url: str):
    """
    Trích xuất kích thước (width x height) từ URL theo định dạng '300x200'.
    Nếu không tìm thấy, trả về kích thước mặc định (100, 100).
    """
    matches = re.findall(r"(\d+)x(\d+)", url)
    if matches:
        width, height = matches[0]
        return int(width), int(height)
    else:
        return (100, 100)


def create_alt_url_mapping(code: str) -> Dict[str, str]:
    """
    Tạo mapping từ thuộc tính alt đến URL của các ảnh trong HTML.
    Chỉ mapping những ảnh không có URL bắt đầu bằng https://placehold.co.
    """
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")

    mapping: Dict[str, str] = {}
    for image in images:
        if not image["src"].startswith("https://placehold.co"):
            mapping[image["alt"]] = image["src"]

    return mapping


async def generate_images(
    code: str,
    api_key: str | None,
    base_url: Union[str, None],
    image_cache: Dict[str, str],
    azure_openai_api_key: str | None = None,
    azure_openai_dalle3_api_version: str | None = None,
    azure_openai_resource_name: str | None = None,
    azure_openai_dalle3_deployment_name: str | None = None,
    model: Literal["dalle3", "flux"] = "dalle3",
) -> str:
    """
    Quét HTML để tìm các thẻ <img> có src là placeholder (https://placehold.co),
    sau đó tạo ảnh mới dựa trên alt text và thay thế URL cũ bằng URL của ảnh được tạo.
    
    Hỗ trợ cả trường hợp sử dụng OpenAI thông thường và Azure OpenAI.
    """
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")

    # Lấy danh sách alt text của các ảnh cần thay thế (chỉ lấy những ảnh chưa có trong cache)
    alts: List[str | None] = []
    for img in images:
        if (
            img["src"].startswith("https://placehold.co")
            and image_cache.get(img.get("alt")) is None
        ):
            alts.append(img.get("alt", None))

    # Loại bỏ giá trị None và lặp lại (unique)
    filtered_alts: List[str] = [alt for alt in alts if alt is not None]
    prompts = list(set(filtered_alts))

    # Nếu không có ảnh cần thay thế, trả về code ban đầu
    if len(prompts) == 0:
        return code

    results = await process_tasks(
        prompts,
        api_key,
        base_url,
        azure_openai_api_key,
        azure_openai_dalle3_api_version,
        azure_openai_resource_name,
        azure_openai_dalle3_deployment_name,
        model,
    )

    # Tạo mapping từ alt text sang URL ảnh được tạo
    mapped_image_urls = dict(zip(prompts, results))
    mapped_image_urls = {**mapped_image_urls, **image_cache}

    # Thay thế URL ảnh trong HTML
    for img in images:
        if not img["src"].startswith("https://placehold.co"):
            continue

        new_url = mapped_image_urls.get(img.get("alt"))
        if new_url:
            width, height = extract_dimensions(img["src"])
            img["width"] = width
            img["height"] = height
            img["src"] = new_url
        else:
            print("Image generation failed for alt text: " + str(img.get("alt")))
    return soup.prettify()