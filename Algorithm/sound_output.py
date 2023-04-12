"""
This code is to be run on the local laptop environment.
It will receive sound bytes from the code running on AWS instance and output it using PyAudio library.
"""


# for sound
import pyaudio

# import AWS SDK
import boto3

# Set up the Kinesis Data Stream client
kds_client = boto3.client('kinesis', region_name='ap-southeast-1')

shard_id = 'shardId-000000000000'
partition_key = 'partitionkey'

# Get the latest shard iterator as we want real-time info
response = kds_client.get_shard_iterator(
    StreamName='RenaissanceCapstone',
    ShardId=shard_id,
    ShardIteratorType='LATEST'
)
shard_iterator = response['ShardIterator']



# Initlialise pyaudio stream & associated constants
p = pyaudio.PyAudio()
FS = 44100  # sampling rate, Hz, must be integer
CHANNELS = 2
FORMAT = pyaudio.paFloat32
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=FS,
                output=True)



# Continuously listen for new records
while True:
    # Get the next record
    response = kds_client.get_records(
        ShardIterator=shard_iterator,
        Limit=1
    )
    
    # Process the record if it exists
    if len(response['Records']) > 0:
        data_bytes = response['Records'][0]['Data']
        stream.write(data_bytes)
        print(data_bytes)
        
        # Update the shard iterator for the next record
        shard_iterator = response['NextShardIterator']