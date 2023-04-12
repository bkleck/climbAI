//
//  WorkoutManager.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import HealthKit
import Foundation

class WorkoutManager: NSObject, ObservableObject {
    
    var climbWorkout: HKWorkoutActivityType = .climbing
    
    var selectedWorkout: HKWorkoutActivityType?{
        didSet{
//            print("what is my workout", selectedWorkout?.name)
            guard let selectedWorkout = selectedWorkout else {return}
            startWorkout(workoutType: selectedWorkout)
        }
    }
    
    @Published var showingSummaryView: Bool = false{
        didSet{
            // Sheet dismissed
            if showingSummaryView == false{
                resetWorkout()
                
            }
        }
    }
    
    
    let healthStore = HKHealthStore()
    var session: HKWorkoutSession?
    var builder: HKLiveWorkoutBuilder?
    
    func startWorkout(workoutType: HKWorkoutActivityType){
        let configuration = HKWorkoutConfiguration()
        configuration.activityType = workoutType
        configuration.locationType = .outdoor
        
        do {
            session = try HKWorkoutSession(healthStore: healthStore, configuration: configuration)
            builder = session?.associatedWorkoutBuilder()
        }catch{
            return
        }
        
        builder?.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore, workoutConfiguration: configuration)
        
        session?.delegate = self
        builder?.delegate = self
        
        // start workout session and begin data collection
        let startDate = Date()
        session?.startActivity(with: startDate)
        builder?.beginCollection(withStart: startDate){
            (success, error) in
            // The workout has started
        }
    }
    
    // Request authorization to access healthkit
    
    func requestAuthorization(){
        
        let allTypes = Set([HKObjectType.workoutType(),
                            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
                            HKObjectType.quantityType(forIdentifier: .heartRate)!])
        
        
        healthStore.requestAuthorization(toShare: allTypes, read: allTypes) { (success, error) in
            if !success {
                // Handle the error here.
            }
        }
    }
    
    // State Control
    
    @Published var running = false
    
    func pause(){
        debugPrint("pause")
        session?.pause()
    }
    
    func resume(){
        debugPrint("resume")
        session?.resume()
    }
    
    func togglePause(){
        if running == true{
            pause()
        }else{
            resume()
        }
    }
    
    func endWorkout(){
        debugPrint("end")
        session?.end()
        selectedWorkout=nil
        showingSummaryView = true

    }
    
    @Published var averageHeartRate: Double = 0
    @Published var heartRate: Double = 0
    @Published var activeEnergy: Double = 0
    @Published var workout: HKWorkout?
    
    func updateForStatistics(_ statistics: HKStatistics?){
        guard let statistics = statistics else {return}
        
        DispatchQueue.main.async {
            switch statistics.quantityType{
                case HKQuantityType.quantityType(forIdentifier: .heartRate):
                    let heartRateUnit = HKUnit.count().unitDivided(by: HKUnit.minute())
                    self.heartRate = statistics.mostRecentQuantity()?.doubleValue(for: heartRateUnit) ?? 0
                    self.averageHeartRate = statistics.averageQuantity()?.doubleValue(for: heartRateUnit) ?? 0.0
                case HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned):
                    let energyUnit = HKUnit.smallCalorie()
//                    self.activeEnergy = statistics.sumQuantity()?.doubleValue(for: energyUnit) ?? 0.0
                    self.activeEnergy = statistics.sumQuantity()?.doubleValue(for: energyUnit) ?? 0.0
                default:
                    return
            }
        }
    }
    
    func resetWorkout(){
        selectedWorkout = nil
        builder = nil
        session = nil
        workout = nil
        activeEnergy = 0
        averageHeartRate = 0
        heartRate = 0
    }
    
    
}

// HKWorkoutSessionDelegate
extension WorkoutManager: HKWorkoutSessionDelegate{
    func workoutSession(_ workoutSession: HKWorkoutSession, didFailWithError error: Error) {
        
    }
    
    func workoutSession(_ workoutSession: HKWorkoutSession, didChangeTo toState: HKWorkoutSessionState, from fromState: HKWorkoutSessionState, date: Date) {
        DispatchQueue.main.async{
            self.running = toState == .running
        }
        
        // Wait for session to transition states before ending the builder
        if toState == .ended{
            builder?.endCollection(withEnd: date){
                (session,error) in self.builder?.finishWorkout{
                    (workout, error) in
                    DispatchQueue.main.async{
                        self.workout = workout
                    }
                }
            }
        }
        
    }
}

extension WorkoutManager: HKLiveWorkoutBuilderDelegate{
    func workoutBuilder(_ workoutBuilder: HKLiveWorkoutBuilder, didCollectDataOf collectedTypes: Set<HKSampleType>) {
        for type in collectedTypes{
//            guard let quantityType = type as ? HKQuantityType else {return}
            
            let quantityType:HKQuantityType = type as! HKQuantityType
            let statistics = workoutBuilder.statistics(for: quantityType)
            
            updateForStatistics(statistics)
        }
    }
    
    func workoutBuilderDidCollectEvent(_ workoutBuilder: HKLiveWorkoutBuilder) {
        
    }
    
    
}
