import axios from 'axios'
import React from 'react'

function ImageItem(props) {

    return (
        <div>
            <p>
                <span style={{ fontWeight: 'bold, underline' }}>{props.todo.title} : </span> {props.todo.description} 
                <hr></hr>
            </p>
        </div>
    )
}

export default ImageItem;