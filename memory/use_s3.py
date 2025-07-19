# 업로드
import boto3

s3 = boto3.client('s3')
bucket_name = 'story-gen-memory-store'

def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, bucket_name, s3_key)
    print(f"✅ Uploaded {local_path} to s3://{bucket_name}/{s3_key}")

# 다운로드
def download_from_s3(s3_key, local_path):
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"✅ Downloaded s3://{bucket_name}/{s3_key} to {local_path}")

