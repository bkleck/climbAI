//
//  AudioManager.swift
//  MyWorkouts Watch App
//
//  Created by Xinrui gao on 18/1/23.
//

import Foundation
import AVFoundation

final class AudioManager{
    static let shared = AudioManager()
    
    var player: AVAudioPlayer?
    
    func startPlayer(track: String){
        guard let url = Bundle.main.url(forResource: track, withExtension: "mp3") else {
            print("Resource not found: \(track)")
            return
        }
        
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
        }catch{
            print("failed to init player",error)
        }
    }
}
