//
//  ElapsedTimeView.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 17/1/23.
//

import SwiftUI

struct ElapsesTimeView: View {
    var elapsedTime: TimeInterval = 0
    var showSubseconds: Bool = true
    
    @State private var timeFormatter = ElapsedTimeFormatter()
    
    var body: some View {
        Text(NSNumber(value: elapsedTime), formatter: timeFormatter).fontWeight(.semibold).onChange(of: showSubseconds){
            timeFormatter.showSubseconds = $0
        }
    }
}

struct ElapsesTimeView_Previews: PreviewProvider {
    static var previews: some View {
        ElapsesTimeView()
    }
}

class ElapsedTimeFormatter: Formatter {
    let componentsFormatter : DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.zeroFormattingBehavior = .pad
        return formatter
    }()
    
    var showSubseconds = true
    
    override func string(for value: Any?) -> String? {
        guard let time = value as? TimeInterval else {
            return nil
        }
        
        guard let formattedString = componentsFormatter.string(from: time)else{
            return nil
        }
        
        if showSubseconds{
            let hundreths = Int((time.truncatingRemainder(dividingBy: 1)) * 100)
            let decimalSeparator = Locale.current.decimalSeparator ?? "."
            return String(format: "%@%@%.2d", formattedString, decimalSeparator, hundreths)
            
        }
        return formattedString
    }
}
