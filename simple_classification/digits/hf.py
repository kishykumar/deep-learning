from huggingface_hub import HfApi

api = HfApi()

try:
    url = api.upload_file(
        path_or_fileobj="mnist_cnn.pth",
        path_in_repo="mnist_cnn.pth",
        repo_id="touchingface/digits",
        repo_type="space",
    )
    print("Upload succeeded:", url)
except Exception as e:
    print("Upload failed:", e)