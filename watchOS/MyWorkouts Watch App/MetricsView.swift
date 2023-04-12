//
//  MetricsView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI

struct MetricsView: View {
    
    @EnvironmentObject var workoutManager: WorkoutManager
    
    var body: some View {
        
        TimelineView(.animation) {
            context in
            VStack(alignment: .leading){
//                Text(workoutManager.selectedWorkout?.name ?? "nothing")
                ElapsesTimeView(elapsedTime: workoutManager.builder?.elapsedTime ?? 0.0, showSubseconds: context.cadence == .live).foregroundColor(Color.yellow)
                Text(Measurement(value: workoutManager.activeEnergy , unit: UnitEnergy.calories).formatted(
                    .measurement(width: .abbreviated, usage: .workout, numberFormatStyle: .number))
                )
                Text(workoutManager.heartRate.formatted(.number.precision(.fractionLength(0))) + " bpm")
            }
            .font(.system(.title, design: .rounded).monospacedDigit().lowercaseSmallCaps())
            .frame(maxWidth: .infinity, alignment: .leading)
            .ignoresSafeArea(edges: .bottom)
            .scenePadding()
        }
        
    }
}

struct MetricsView_Previews: PreviewProvider {
    static var previews: some View {
        MetricsView()
    }
}

    
