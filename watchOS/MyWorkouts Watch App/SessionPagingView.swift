//
//  SessionPagingView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI

struct SessionPagingView: View {
    
    @EnvironmentObject var workoutManager: WorkoutManager
    
    @State private var selection: Tab = .metrics
    
    enum Tab{
        case controls, metrics
    }
    
    var body: some View {
        TabView(selection: $selection){
            ControlsView().tag(Tab.controls)
            MetricsView().tag(Tab.metrics)
        }
        .navigationTitle(workoutManager.selectedWorkout?.name ?? "")
        .navigationBarBackButtonHidden(true)
        .onChange(of: workoutManager.running) {
            _ in displayMetricsView()
        }
    }
    
    private func displayMetricsView(){
        withAnimation{
            selection = .metrics
        }
    }
}

struct SessionPagingView_Previews: PreviewProvider {
    static var previews: some View {
        SessionPagingView()
    }
}
