package com.pedro.rtpstreamer.defaultexample;

import android.util.Log;

import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.regions.Region;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.sqs.AmazonSQS;
import com.amazonaws.services.sqs.AmazonSQSClient;
import com.amazonaws.services.sqs.model.DeleteMessageRequest;
import com.amazonaws.services.sqs.model.Message;
import com.amazonaws.services.sqs.model.ReceiveMessageRequest;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class SQSClient {
    private AmazonSQS sqsClient;

    public SQSClient() {
        // Replace ACCESS_KEY_ID and SECRET_ACCESS_KEY with your own AWS access key ID and secret access key
        BasicAWSCredentials credentials = new BasicAWSCredentials("AKIAW25K4CST5QRYJ5A6", "yDWDuAwarmqzRjy0dY7TsnQEf1wC/2k93NLW8Of8");

        // Initialize Amazon SQS client
        sqsClient = new AmazonSQSClient(credentials);
        sqsClient.setRegion(Region.getRegion(Regions.AP_SOUTHEAST_1.getName()));
        Logger.getLogger("com.amazonaws.auth.AWS4Signer").setLevel(Level.SEVERE);


    }

    public String readFromQueue() {
        String queueUrl = "https://sqs.ap-southeast-1.amazonaws.com/470120404135/RenaissanceCapstone";
        ReceiveMessageRequest receiveMessageRequest = new ReceiveMessageRequest(queueUrl)
                .withWaitTimeSeconds(0)  // Wait for up to 0 seconds for a message to be available
                .withMaxNumberOfMessages(1);  // Receive up to 1 messages at once

        List<Message> messages = sqsClient.receiveMessage(receiveMessageRequest).getMessages();

        if (!messages.isEmpty()) {
            Message message = messages.get(0);
            String body = message.getBody();
            System.out.println("Received message: " + body);

            // Extract the message value
            String messageValue = "";
            try {
                JSONObject jsonMessage = new JSONObject(body);
                messageValue = jsonMessage.getString("value");
            } catch (JSONException e) {
                System.out.println("Error parsing JSON message: " + e.getMessage());
            }
            System.out.println("Message value: " + messageValue);

            // Delete the message from the queue
            String messageReceiptHandle = message.getReceiptHandle();
            sqsClient.deleteMessage(new DeleteMessageRequest(queueUrl, messageReceiptHandle));
            System.out.println("Deleted message: " + body);

            // Return the message value
            return body;
        }
        return null;
    }
}
