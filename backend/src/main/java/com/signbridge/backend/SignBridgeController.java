
package com.signbridge.backend;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class SignBridgeController {

    @GetMapping("/api/test")
    public Map<String, String> test() {
        return Map.of(
                "project", "SignBridge",
                "status", "Backend is working!"
        );
    }
}
