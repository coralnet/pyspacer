import boto3

client = boto3.client('batch')

resp = client.list_jobs(
    jobQueue='production',
    jobStatus='SUCCEEDED'
)
print(len(resp['jobSummaryList']))
