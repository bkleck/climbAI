//
//  ControlsView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI
import AVFoundation


struct ControlsView: View {
    
    @EnvironmentObject var workoutManager: WorkoutManager
    
    @State var data: [DataModel] = []
    
    @State var idToModify = ""
    @State var colorToModify = ""
    @State var dateToModify = ""
    
    @State var toModify: DataModel = DataModel(
        id: "",
        date: "",
        timeInSeconds: "",
        calories: "",
        color: "",
        heartRate: "")

    @State var isRunning: Bool = false
    
    @State var responseData: String = ""
    
    @State var jsonString: String = ""

    
    let syn = AVSpeechSynthesizer()
    
    func generateRandomNumber() -> String {
        let randomNumber = Int.random(in: 10000000...99999999)
        return String(randomNumber)
    }
    
    func findObjectWithEmptyHeartRate(exercises: [DataModel]) -> DataModel? {
        for item in exercises {
            if item.heartRate == "empty" {
                print("empty item")
                print(item)
                return item
            }
        }
        return DataModel(
            id: "",
            date: "",
            timeInSeconds: "",
            calories: "",
            color: "",
            heartRate: "")
    }
    
    


    var body: some View {
        
        HStack{
//            VStack{
//                Button{
//                    workoutManager.endWorkout()
//                } label: {
//                    Image(systemName: "xmark")
//                }
//                .tint(Color.red)
//                .font(.title2)
//                Text("End")
//            }
            VStack{
                Button{
                    
                    //GET API
                    guard let url = URL(string: "https://4hc9b6oi3f.execute-api.ap-southeast-1.amazonaws.com/Prod/getalldynamodb") else {
                        return
                    }

                    let task = URLSession.shared.dataTask(with: url) { data, response, error in
                        guard let data = data, error == nil else {
                            return
                        }
                        do {
                            
                            let json = try JSONSerialization.jsonObject(with: data, options: [])
                            if let dict = json as? [String: Any], let body = dict["body"] as? String {
                                print("Body: \(body)")
                                
                                let jsonData = body.data(using: .utf8)!
                                print("JsonDATA----->")
                                print(jsonData)
                                let decoder = JSONDecoder()
                                if let decodedData = try? decoder.decode([DataModel].self, from: jsonData) {
                                    self.data = decodedData
                                    print("decoded data")
                                    print(decodedData)
                                    
                                    let exercise = findObjectWithEmptyHeartRate(exercises: decodedData)
                                    
                                    self.idToModify = exercise!.id
                                    self.colorToModify = exercise!.color
                                    self.dateToModify = exercise!.date
                                    
                                    //populate with health stats + time
                                    self.toModify.id = self.idToModify
                                    self.toModify.color = self.colorToModify
                                    self.toModify.date = self.dateToModify
                                    
                                    self.toModify.timeInSeconds = String(workoutManager.builder?.elapsedTime ?? 0.0)
                                    
                                    self.toModify.calories = String(workoutManager.activeEnergy)
                                    
                                    self.toModify.heartRate = String(workoutManager.heartRate.formatted(.number.precision(.fractionLength(0))))
                                    
                                    print("tomodify")
                                    print(self.toModify)
                                    
                                    let encoder = JSONEncoder()
                                    encoder.outputFormatting = .prettyPrinted

                                    guard let data = try? encoder.encode(self.toModify),
                                          let dictionary = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
                                        print("Failed to convert object to dictionary")
                                        return
                                    }
                                    
                                    guard let jsonData = try? JSONSerialization.data(withJSONObject: dictionary, options: []),
                                          let jsonString = String(data: jsonData, encoding: .utf8) else {
                                        print("Failed to convert dictionary to string")
                                        return
                                    }
                                    self.jsonString = jsonString
                                    
                                    
                                    print("JSON String ---- >")
                                    print(self.jsonString)
                                    
                                    
                                    // PUT REQUEST
                                    guard let url = URL(string: "https://s2rjzbmgzd.execute-api.ap-southeast-1.amazonaws.com/Prod/putdynamodb") else {
                                        print("Invalid URL")
                                        return
                                    }
                                    
                                    var putRequest = URLRequest(url: url)
                                    putRequest.httpMethod = "PUT"
                                    
                                    // Set the request body if needed
                                    let requestBody = self.jsonString
                                    print("REQUEST BODY")
                                    print(requestBody)
                                    putRequest.httpBody = requestBody.data(using: .utf8)
                                    
                                    // Send the request
                                    let session = URLSession.shared
                                    let putTask = session.dataTask(with: putRequest) { data, response, error in
                                        if let error = error {
                                            print("Error: \(error)")
                                            return
                                        }
                                        
                                        if let httpResponse = response as? HTTPURLResponse {
                                            print("Status code: \(httpResponse.statusCode)")
                                        }
                                        
                                        if let data = data {
                                            // Handle the response data
                                            print("Response data: \(String(data: data, encoding: .utf8) ?? "")")
                                        }
                                    }
                                    
                                    putTask.resume()
                                    
                                    // PUT REQUEST
                                    
                                }
                            }
                        } catch {
                            print(error.localizedDescription)
                        }
                    }
                    task.resume()
                    //GET API
                    
                    /*
                    // PUT REQUEST
                    guard let url = URL(string: "https://s2rjzbmgzd.execute-api.ap-southeast-1.amazonaws.com/Prod/putdynamodb") else {
                        print("Invalid URL")
                        return
                    }
                    
                    var putRequest = URLRequest(url: url)
                    putRequest.httpMethod = "PUT"
                    
                    // Set the request body if needed
                    let requestBody = self.jsonString
                    print("REQUEST BODY")
                    print(requestBody)
                    putRequest.httpBody = requestBody.data(using: .utf8)
                    
                    // Send the request
                    let session = URLSession.shared
                    let putTask = session.dataTask(with: putRequest) { data, response, error in
                        if let error = error {
                            print("Error: \(error)")
                            return
                        }
                        
                        if let httpResponse = response as? HTTPURLResponse {
                            print("Status code: \(httpResponse.statusCode)")
                        }
                        
                        if let data = data {
                            // Handle the response data
                            print("Response data: \(String(data: data, encoding: .utf8) ?? "")")
                        }
                    }
                    
                    putTask.resume()
                    //PUT REQUEST
                     */

                    /*
                            
                    // Create the URL for the JSONPlaceholder API endpoint
                    guard let url = URL(string: "https://mzpoqw4tt9.execute-api.ap-southeast-1.amazonaws.com/Prod/applewatchstats") else { return }

                    // Create the URL request and configure it with the appropriate method (e.g., GET, POST, etc.)
                    var putRequest = URLRequest(url: url)
                    putRequest.httpMethod = "POST"
                    putRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    // Create a dictionary to hold the data to be sent in the request body
                    let postData = [ "id": generateRandomNumber(),
                                     "color":"",
                                     "timeInSeconds":String(workoutManager.builder?.elapsedTime ?? 0.0),
                                     "date":"",
                                     "calories":String(workoutManager.activeEnergy),
                                     "heartRate":String(workoutManager.heartRate.formatted(.number.precision(.fractionLength(0)))),
                                     "createdAt":"2022-02-26T16:37:48.244Z",
                                     "updatedAt":"2022-02-26T16:37:48.244Z"] as [String : Any]

                    // Convert the data to JSON format and add it to the request body
                    guard let httpBody = try? JSONSerialization.data(withJSONObject: postData, options: []) else { return }
                    putRequest.httpBody = httpBody

                    // Create a URLSession and use it to send the request to the API endpoint
                    let putSession = URLSession.shared
                    putSession.dataTask(with: request) { data, response, error in
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
                    
                    print("WORKOUT STATS")
                    print(workoutManager.builder?.elapsedTime ?? 0.0)
                    print(workoutManager.activeEnergy)
                    print(workoutManager.heartRate.formatted(.number.precision(.fractionLength(0))))
                     
                     */
                    
                    
                    AudioManager.shared.startPlayer(track: "w_end")
                    
                    workoutManager.togglePause()
                    isRunning = !isRunning
                    

                } label: {
                    Image(systemName: "xmark").resizable(resizingMode: .stretch)
                        .font(.largeTitle)
                        .frame(width: 88, height: 88)
                }
                .tint(Color.red)
                .font(.title2)
                
            }
        }
    }
}

struct ControlsView_Previews: PreviewProvider {
    static var previews: some View {
        ControlsView()
    }
}

struct DataModel: Identifiable,Encodable, Decodable {
    var id: String
    var date: String
    var timeInSeconds: String
    var calories: String
    var color: String
    var heartRate: String
}


