//
//  SummaryView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI
import HealthKit

struct SummaryView: View {
    
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var workoutManager: WorkoutManager
    
    @State private var durationFormatter:
    DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter
    }()
    var body: some View {
        if workoutManager.workout == nil{
            ProgressView("Saving Workout").navigationBarHidden(true)
        }else{
            ScrollView(.vertical){
                VStack(alignment: .leading){
                    SummaryMetricView(title: "Total Time", value: durationFormatter.string(from: 30 * 60 + 15) ?? "").accentColor(Color.yellow)
                    
                    SummaryMetricView(
                        title: "Total Energy",
                        value: Measurement(value: 96, unit: UnitEnergy.kilocalories)
                            .formatted(.measurement(width: .abbreviated, usage: .workout, numberFormatStyle: .number))).accentColor(Color.green)
                    
                    SummaryMetricView(title: "Avg. Heart Rate", value: 142.formatted(.number.precision(.fractionLength(0))) + "bpm").accentColor(Color.red)
                    
                    Button("Done"){
                        exit(0)
                    }
                }
                .scenePadding()
                
            }
            .navigationTitle("Summary")
            .navigationBarTitleDisplayMode(.inline)
        }
        
    }
    
}

struct SummaryView_Previews: PreviewProvider {
    static var previews: some View {
        SummaryView()
    }
}

struct SummaryMetricView: View {
    var title: String
    var value: String
    
    var body: some View {
        Text(title)
        Text(value)
            .font(.system(.title2, design: .rounded).lowercaseSmallCaps()).foregroundColor(.accentColor)
        Divider()
    }
}
