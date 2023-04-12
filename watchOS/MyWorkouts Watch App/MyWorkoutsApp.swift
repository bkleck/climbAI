//
//  MyWorkoutsApp.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI

@main
struct MyWorkouts_Watch_AppApp: App {
    
    @StateObject var workoutManager = WorkoutManager()
    
    @SceneBuilder var body: some Scene {
        WindowGroup {
            NavigationView{
                ContentView()
            }.sheet(isPresented: $workoutManager.showingSummaryView){
                SummaryView()
            }
            .environmentObject(workoutManager)
        }
    }
}
