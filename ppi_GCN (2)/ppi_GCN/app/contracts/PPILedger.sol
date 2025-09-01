// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title PPILedger
 * @dev Smart contract for logging PPI prediction and drug discovery events
 */
contract PPILedger {
    address public owner;
    uint256 public eventCount;
    
    struct LogEntry {
        uint256 id;
        address user;
        string action;
        string refId;
        string metadata;
        uint256 timestamp;
        bytes32 txHash;
    }
    
    // Events
    event Logged(
        uint256 indexed id,
        address indexed user,
        string action,
        string refId,
        uint256 timestamp
    );
    
    event UserRegistered(
        address indexed user,
        string institution,
        uint256 timestamp
    );
    
    event ModelTrained(
        address indexed user,
        string modelType,
        string jobId,
        uint256 timestamp
    );
    
    event PredictionMade(
        address indexed user,
        string predictionId,
        string proteinPair,
        uint256 timestamp
    );
    
    event DrugSuggested(
        address indexed user,
        string targetProtein,
        string drugId,
        uint256 timestamp
    );
    
    event DataSynced(
        address indexed user,
        string dataSource,
        uint256 recordCount,
        uint256 timestamp
    );
    
    // Mappings
    mapping(uint256 => LogEntry) public logs;
    mapping(address => bool) public authorizedUsers;
    mapping(string => uint256) public actionCounts;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedUsers[msg.sender] || msg.sender == owner, "Not authorized");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        authorizedUsers[msg.sender] = true;
        eventCount = 0;
    }
    
    /**
     * @dev Authorize a user to log events
     */
    function authorizeUser(address user) external onlyOwner {
        authorizedUsers[user] = true;
    }
    
    /**
     * @dev Revoke user authorization
     */
    function revokeUser(address user) external onlyOwner {
        authorizedUsers[user] = false;
    }
    
    /**
     * @dev Log a general event
     */
    function logEvent(
        address user,
        string memory action,
        string memory refId,
        string memory metadata
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: action,
            refId: refId,
            metadata: metadata,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts[action]++;
        
        emit Logged(eventCount, user, action, refId, block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Log user registration
     */
    function logUserRegistration(
        address user,
        string memory institution
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: "auth.register",
            refId: "registration",
            metadata: institution,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts["auth.register"]++;
        
        emit UserRegistered(user, institution, block.timestamp);
        emit Logged(eventCount, user, "auth.register", "registration", block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Log model training event
     */
    function logModelTraining(
        address user,
        string memory modelType,
        string memory jobId,
        string memory config
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        string memory action = string(abi.encodePacked("models.train.", modelType));
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: action,
            refId: jobId,
            metadata: config,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts[action]++;
        
        emit ModelTrained(user, modelType, jobId, block.timestamp);
        emit Logged(eventCount, user, action, jobId, block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Log PPI prediction
     */
    function logPPIPrediction(
        address user,
        string memory predictionId,
        string memory proteinA,
        string memory proteinB,
        string memory scores
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        string memory proteinPair = string(abi.encodePacked(proteinA, ":", proteinB));
        string memory metadata = string(abi.encodePacked(
            '{"proteins":"', proteinPair, '","scores":', scores, '}'
        ));
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: "ppi.predict",
            refId: predictionId,
            metadata: metadata,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts["ppi.predict"]++;
        
        emit PredictionMade(user, predictionId, proteinPair, block.timestamp);
        emit Logged(eventCount, user, "ppi.predict", predictionId, block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Log drug suggestion
     */
    function logDrugSuggestion(
        address user,
        string memory targetProtein,
        string memory drugId,
        string memory confidence
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        string memory metadata = string(abi.encodePacked(
            '{"target":"', targetProtein, '","drug":"', drugId, '","confidence":', confidence, '}'
        ));
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: "drug.suggest",
            refId: drugId,
            metadata: metadata,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts["drug.suggest"]++;
        
        emit DrugSuggested(user, targetProtein, drugId, block.timestamp);
        emit Logged(eventCount, user, "drug.suggest", drugId, block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Log data synchronization
     */
    function logDataSync(
        address user,
        string memory dataSource,
        uint256 recordCount
    ) external onlyAuthorized returns (uint256) {
        eventCount++;
        
        string memory action = string(abi.encodePacked("drugs.sync.", dataSource));
        string memory metadata = string(abi.encodePacked(
            '{"source":"', dataSource, '","records":', uintToString(recordCount), '}'
        ));
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: action,
            refId: dataSource,
            metadata: metadata,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts[action]++;
        
        emit DataSynced(user, dataSource, recordCount, block.timestamp);
        emit Logged(eventCount, user, action, dataSource, block.timestamp);
        
        return eventCount;
    }
    
    /**
     * @dev Get logs by user
     */
    function getLogsByUser(address user, uint256 offset, uint256 limit) 
        external view returns (LogEntry[] memory) {
        uint256 count = 0;
        
        // Count matching logs
        for (uint256 i = 1; i <= eventCount; i++) {
            if (logs[i].user == user) {
                count++;
            }
        }
        
        // Handle pagination
        if (offset >= count) {
            return new LogEntry[](0);
        }
        
        uint256 end = offset + limit;
        if (end > count) {
            end = count;
        }
        
        LogEntry[] memory result = new LogEntry[](end - offset);
        uint256 resultIndex = 0;
        uint256 userLogIndex = 0;
        
        for (uint256 i = 1; i <= eventCount && resultIndex < (end - offset); i++) {
            if (logs[i].user == user) {
                if (userLogIndex >= offset) {
                    result[resultIndex] = logs[i];
                    resultIndex++;
                }
                userLogIndex++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Get logs by action type
     */
    function getLogsByAction(string memory action, uint256 offset, uint256 limit) 
        external view returns (LogEntry[] memory) {
        uint256 count = 0;
        
        // Count matching logs
        for (uint256 i = 1; i <= eventCount; i++) {
            if (keccak256(bytes(logs[i].action)) == keccak256(bytes(action))) {
                count++;
            }
        }
        
        // Handle pagination
        if (offset >= count) {
            return new LogEntry[](0);
        }
        
        uint256 end = offset + limit;
        if (end > count) {
            end = count;
        }
        
        LogEntry[] memory result = new LogEntry[](end - offset);
        uint256 resultIndex = 0;
        uint256 actionLogIndex = 0;
        
        for (uint256 i = 1; i <= eventCount && resultIndex < (end - offset); i++) {
            if (keccak256(bytes(logs[i].action)) == keccak256(bytes(action))) {
                if (actionLogIndex >= offset) {
                    result[resultIndex] = logs[i];
                    resultIndex++;
                }
                actionLogIndex++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Get recent logs
     */
    function getRecentLogs(uint256 limit) external view returns (LogEntry[] memory) {
        if (limit > eventCount) {
            limit = eventCount;
        }
        
        LogEntry[] memory result = new LogEntry[](limit);
        uint256 startIndex = eventCount - limit + 1;
        
        for (uint256 i = 0; i < limit; i++) {
            result[i] = logs[startIndex + i];
        }
        
        return result;
    }
    
    /**
     * @dev Get action statistics
     */
    function getActionCount(string memory action) external view returns (uint256) {
        return actionCounts[action];
    }
    
    /**
     * @dev Get contract statistics
     */
    function getStats() external view returns (
        uint256 totalEvents,
        uint256 totalUsers,
        uint256 timestamp
    ) {
        // Count unique users
        address[] memory users = new address[](eventCount);
        uint256 uniqueUsers = 0;
        
        for (uint256 i = 1; i <= eventCount; i++) {
            bool isNew = true;
            for (uint256 j = 0; j < uniqueUsers; j++) {
                if (users[j] == logs[i].user) {
                    isNew = false;
                    break;
                }
            }
            if (isNew) {
                users[uniqueUsers] = logs[i].user;
                uniqueUsers++;
            }
        }
        
        return (eventCount, uniqueUsers, block.timestamp);
    }
    
    /**
     * @dev Utility function to convert uint to string
     */
    function uintToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        
        uint256 temp = value;
        uint256 digits;
        
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        
        bytes memory buffer = new bytes(digits);
        
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        
        return string(buffer);
    }
    
    /**
     * @dev Emergency pause function
     */
    bool public paused = false;
    
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    function pause() external onlyOwner {
        paused = true;
    }
    
    function unpause() external onlyOwner {
        paused = false;
    }
    
    /**
     * @dev Override logEvent to include pause check
     */
    function logEvent(
        address user,
        string memory action,
        string memory refId,
        string memory metadata
    ) external onlyAuthorized whenNotPaused returns (uint256) {
        eventCount++;
        
        LogEntry memory entry = LogEntry({
            id: eventCount,
            user: user,
            action: action,
            refId: refId,
            metadata: metadata,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        });
        
        logs[eventCount] = entry;
        actionCounts[action]++;
        
        emit Logged(eventCount, user, action, refId, block.timestamp);
        
        return eventCount;
    }
}