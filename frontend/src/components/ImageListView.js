import ImageItem from './Image'

export default function ImageView(props) {
    return (
        <div>
            <ul>
                {props.imageList.map(image => <ImageItem image={image} />)}
            </ul>
        </div>
    )
}