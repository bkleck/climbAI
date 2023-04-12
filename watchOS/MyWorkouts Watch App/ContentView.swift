//
//  ContentView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI
import HealthKit
import AVFoundation

struct ContentView: View {
    
    @State private var textToSpeak = "Hello, this is a sample text to speak."
    
    @EnvironmentObject var workoutManager: WorkoutManager
    
    @State var responseData: String = ""
    
    var climbWorkout: HKWorkoutActivityType = .running
    
    let syn = AVSpeechSynthesizer()
    
    var body: some View {
        
        VStack{
            NavigationStack{
                
                //                Image(systemName: "headphones")
                //                    .font(.system(size: 30))
                
                
                //                Text("Connect Bluetooth headphones to your AI climb assistant")
                //                    .font(.footnote)
                //                    .multilineTextAlignment(.center)
                
                NavigationLink(destination: SessionPagingView().onAppear {
                    
                    let utterance = AVSpeechUtterance(string: "Workout started")
                    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                    utterance.rate = 0.5
                    
                    syn.speak(utterance)
                    
                    // Define the URL and create a URLRequest
                    guard let url = URL(string: "https://snac8c4owg.execute-api.ap-southeast-1.amazonaws.com/Prod") else {
                        print("Invalid URL")
                        return
                    }
                    var request = URLRequest(url: url)

                    // Set the HTTP method and headers
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    // Define the request body data
                    let data = ["key1": "value1", "key2": "value2"]
                    let jsonData = try? JSONSerialization.data(withJSONObject: data)

                    // Set the request body
                    request.httpBody = jsonData

                    // Create and send the data task
                    let task = URLSession.shared.dataTask(with: request) { (data, response, error) in
                        if let error = error {
                            print("Error: \(error.localizedDescription)")
                        } else if let data = data, let response = response as? HTTPURLResponse {
                            print("Response status code: \(response.statusCode)")
                            // Process the response data as needed
                        }
                    }
                    task.resume()
                    
                    

                    
                    /*
                    
                    // Create the URL for the JSONPlaceholder API endpoint
                    guard let url = URL(string: "https://mzpoqw4tt9.execute-api.ap-southeast-1.amazonaws.com/Prod/applewatchstats") else { return }

                    // Create the URL request and configure it with the appropriate method (e.g., GET, POST, etc.)
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    // Create a dictionary to hold the data to be sent in the request body
                    let postData = [ "id": "postHTTP",
                                     "color":"placeholder",
                                     "timeInSeconds":String(workoutManager.builder?.elapsedTime ?? 0.0),
                                     "date":"placeholder",
                                     "calories":String(workoutManager.activeEnergy),
                                     "heartRate":String(workoutManager.heartRate.formatted(.number.precision(.fractionLength(0)))),
                                     "createdAt":"createdAt",
                                     "updatedAt":"updatedAt"] as [String : Any]

                    // Convert the data to JSON format and add it to the request body
                    guard let httpBody = try? JSONSerialization.data(withJSONObject: postData, options: []) else { return }
                    request.httpBody = httpBody

                    // Create a URLSession and use it to send the request to the API endpoint
                    let session = URLSession.shared
                    session.dataTask(with: request) { data, response, error in
                        // Handle any errors that occur during the request
                        if let error = error {
                            print("Error: \(error.localizedDescription)")
                            return
                        }

                        // If there is data returned from the request, convert it to a string and print it to the console
                        if let data = data {
                            if let stringData = String(data: data, encoding: .utf8) {
                                print("Response: \(stringData)")
                                DispatchQueue.main.async {
                                    responseData = stringData
                                }
                            }
                        }
                    }.resume()
                     */
                    
                    workoutManager.selectedWorkout = climbWorkout
                }
                ) {
                    
                    Image(systemName: "play.fill").resizable(resizingMode: .stretch)
                        .font(.largeTitle)
                        .frame(width: 88, height: 88)
                    
                }
                
            }
            .environmentObject(workoutManager)
        }.onAppear{
            workoutManager.requestAuthorization()
            let utterance = AVSpeechUtterance(string: "Tap on watch face to start workout")
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = 0.5
            
            syn.speak(utterance)
        }
        
        
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

extension HKWorkoutActivityType: Identifiable {
    public var id: UInt {
        rawValue
    }
    
    var name: String {
        switch self{
            case .running:
                return "climbing"
            default:
                return ""
        }
    }
}
